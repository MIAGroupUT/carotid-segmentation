import warnings
import numpy as np
import torch
from dijkstra3d import dijkstra
from scipy.interpolate import interp1d
from typing import Dict, Any
import pandas as pd
from carotid.utils.transforms import unravel_index
from carotid.utils.errors import NoValidSlice


class CenterlineExtractor:
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.label_list = ["internal", "external"]
        self.side_list = ["left", "right"]

    def remove_common_external_centers(self, label_df: pd.DataFrame) -> pd.DataFrame:
        internal_df = label_df[label_df.label == "internal"]
        external_df = label_df[label_df.label == "external"]

        min_internal_z = int(internal_df.z.min())
        min_external_z = int(external_df.z.min())

        z = max(min_internal_z, min_external_z)
        flag = False
        max_z = int(label_df.z.max())
        label_df.set_index(["label", "z"], inplace=True, drop=True)

        while z <= max_z and not flag:
            try:
                external_center = label_df.loc[("external", z), ["x", "y"]].values
                internal_center = label_df.loc[("internal", z), ["x", "y"]].values
                if (
                    np.linalg.norm(internal_center - external_center)
                    > self.parameters["spatial_threshold"]
                ):
                    flag = True
                else:
                    label_df.drop(("external", z), inplace=True)
            except KeyError:
                flag = True
            z += 1

        label_df.reset_index(inplace=True)

        # Change external to common if below internal carotid
        if min_external_z < min_internal_z:
            for z in range(min_external_z, min_internal_z):
                idx = label_df[label_df.z == z].index.item()
                label_df.loc[idx, "label"] = "internal"

        return label_df

    def compute_uncertainty(
        self, label_df: pd.DataFrame, heatmap_dict: Dict[str, torch.Tensor]
    ) -> pd.DataFrame:

        for idx in label_df.index.values:
            label_name = label_df.loc[idx, "label"]
            label_idx = self.label_list.index(label_name)
            x, y, z = label_df.loc[idx, ["x", "y", "z"]].astype(int)
            label_df.loc[idx, "mean_value"] = heatmap_dict["mean"][
                label_idx, x, y, z
            ].item()
            label_df.loc[idx, "std_value"] = heatmap_dict["std"][
                label_idx, x, y, z
            ].item()
            batch_max_indices = heatmap_dict["max_indices"][:, label_idx, z]
            batch_distances = torch.linalg.norm(
                batch_max_indices - torch.Tensor([x, y]), dim=1
            )
            label_df.loc[idx, "max_distances"] = torch.mean(batch_distances).item()

        return label_df


class OnePassExtractor(CenterlineExtractor):
    """
    Extracts centerline from heatmaps by seeding them once.
    """

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        seedpoints = {
            side: self.get_seedpoints(sample[f"{side}_heatmap"]["mean"])
            for side in self.side_list
        }

        for side in self.side_list:
            centerline_dict = self.get_centerline(
                seedpoints[side], sample[f"{side}_heatmap"]["mean"]
            )
            centerline_df = pd.DataFrame(columns=["label", "x", "y", "z"])
            for label_name in centerline_dict.keys():
                label_df = pd.DataFrame(
                    centerline_dict[label_name], columns=["x", "y", "z"]
                )
                label_df["label"] = label_name
                centerline_df = pd.concat([centerline_df, label_df])
            centerline_df = self.remove_common_external_centers(centerline_df)
            centerline_df.dropna(inplace=True)
            centerline_df = self.compute_uncertainty(
                centerline_df, sample[f"{side}_heatmap"]
            )
            sample[f"{side}_centerline"] = centerline_df

        return sample

    def get_seedpoints(self, heatmap_pt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Find seeds that will be connected by the Dijkstra algorithm.

        Args:
            heatmap_pt: heatmap from which the seeds are extracted.

        Returns:
            dictionary with keys corresponding to labels (internal, external) and items to the array of seeds.
        """
        slice_shape = heatmap_pt[0, :, :, 0].shape
        mask_pt = heatmap_pt > self.parameters["threshold"]

        seeds_dict = dict()

        for label_idx, label_name in enumerate(["internal", "external"]):

            # only determine seed points where the given carotid exceed threshold
            (valid_slice_indices,) = torch.where(mask_pt[label_idx].any(1).any(0))
            if len(valid_slice_indices) == 0:
                raise NoValidSlice(
                    f"No valid slices were found for the {label_name} carotid.\n"
                    f"Please try to lower the threshold "
                    f"(current value: {self.parameters['threshold']})."
                )
            min_slice = valid_slice_indices[0].item()
            max_slice = valid_slice_indices[-1].item()
            steps = torch.arange(min_slice, max_slice, self.parameters["step_size"])

            seeds_dict[label_name] = torch.zeros((len(steps) + 1, 3))

            # First seed on min_slice
            seed = unravel_index(
                torch.argmax(heatmap_pt[label_idx, :, :, min_slice]),
                slice_shape,
            )
            seeds_dict[label_name][0] = torch.Tensor([*seed, min_slice])

            # Seeds in each 3D section
            for i in range(len(steps) - 1):
                heatmap_section = heatmap_pt[label_idx, :, :, steps[i] : steps[i + 1]]
                seed = unravel_index(
                    torch.argmax(heatmap_section), heatmap_section.shape
                )
                seed = torch.Tensor(seed) + torch.Tensor([0, 0, steps[i]])
                seeds_dict[label_name][i + 1] = seed

            # Last seed point on max_slice
            seed = unravel_index(
                torch.argmax(heatmap_pt[label_idx, :, :, max_slice]), slice_shape
            )
            seeds_dict[label_name][-1] = torch.Tensor([*seed, max_slice])

        return seeds_dict

    @staticmethod
    def get_centerline(
        seeds_dict: Dict[str, torch.Tensor], heatmap_pt: torch.Tensor
    ) -> Dict[str, np.array]:
        """
        Connect seeds with Dijkstra algorithm to obtain one center per slice.

        Args:
            seeds_dict: dictionary with keys corresponding to labels (internal, external)
                and items to the indices of seeds.
            heatmap_pt: heatmap from which the seeds are extracted.

        Returns:
            dictionary with keys corresponding to labels (internal, external) and items to the array of centers.
        """
        # cost based on heatmap only
        heatmap_np = heatmap_pt.numpy()
        cost_np = np.max(heatmap_np) - heatmap_np

        paths = {"internal": [], "external": []}
        for label_idx, label_name in enumerate(["internal", "external"]):
            label_cost_np = cost_np[label_idx]
            seeds_np = seeds_dict[label_name].numpy()
            for seed_idx in range(seeds_np.shape[0] - 1):
                # Restrict possible space between seeds only
                seed_cost_np = np.copy(label_cost_np)
                min_seed = seeds_np[seed_idx]
                max_seed = seeds_np[seed_idx + 1]
                seed_cost_np[:, :, int(min_seed[2])] = np.inf
                seed_cost_np[:, :, int(max_seed[2]) + 1 :] = np.inf

                dijkstra_path = dijkstra(
                    seed_cost_np,
                    min_seed,
                    max_seed,
                )
                paths[label_name].append(dijkstra_path)
            paths[label_name] = np.vstack(paths[label_name])

            # resample so we have one center per axial slice
            interp = interp1d(
                paths[label_name][:, 2], paths[label_name][:, [0, 1]].transpose()
            )

            # Interpolate between min_slice and max_slice
            slices = np.arange(seeds_np[0, 2], seeds_np[-1, 2] + 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                paths[label_name] = np.concatenate(
                    [interp(slices.T).T, np.expand_dims(slices, 1)], axis=1
                )
        return paths
