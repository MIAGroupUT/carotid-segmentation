import abc
import numpy as np
from dijkstra3d import dijkstra
from scipy.interpolate import interp1d
from typing import Dict, Any
import pandas as pd


side_list = ["left", "right"]


class CenterlineExtractor:
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    @abc.abstractmethod
    def __call__(
        self, sample: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        pass


class OnePassExtractor(CenterlineExtractor):
    """
    Extracts centerline from heatmaps by seeding them once.
    """

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        seedpoints = {
            side: self.get_seedpoints(sample[f"{side}_heatmap"]) for side in side_list
        }

        for side in side_list:
            centerline_dict = self.get_centerline(
                seedpoints[side], sample[f"{side}_heatmap"]
            )
            centerline_df = pd.DataFrame(columns=["label", "z", "y", "x"])
            for label_name in centerline_dict.keys():
                label_df = pd.DataFrame(
                    centerline_dict[label_name], columns=["z", "y", "x"]
                )
                label_df["label"] = label_name
                centerline_df = pd.concat([centerline_df, label_df])

            centerline_df.reset_index(inplace=True, drop=True)
            sample[f"{side}_centerline"] = centerline_df

        return sample

    def get_seedpoints(self, heatmap_np: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Find seeds that will be connected by the Dijkstra algorithm.

        Args:
            heatmap_np: heatmap from which the seeds are extracted.

        Returns:
            dictionary with keys corresponding to labels (internal, external) and items to the array of seeds.
        """
        # TODO take into image transpose
        slice_shape = heatmap_np[0, 0].shape
        mask_np = heatmap_np > self.parameters["threshold"]

        # only determine seed points where both internal/external carotids exceed threshold
        (valid_slice_indices,) = np.where(
            mask_np.any(axis=(-2, -1)).all(axis=0)
        )  # TODO transpose here
        min_slice = valid_slice_indices[0]
        max_slice = valid_slice_indices[-1]
        steps = np.arange(min_slice, max_slice, self.parameters["step_size"])

        seeds_dict = {
            "internal": np.zeros((len(steps) + 1, 3)),
            "external": np.zeros((len(steps) + 1, 3)),
        }

        for label_idx, label_name in enumerate(["internal", "external"]):
            # First seed on min_slice
            seed = np.unravel_index(
                np.argmax(heatmap_np[label_idx, min_slice]),
                slice_shape,
            )  # TODO transpose here
            seeds_dict[label_name][0] = np.array([min_slice, *seed])

            # Seeds in each 3D section
            for i in range(len(steps) - 1):
                heatmap_section = heatmap_np[label_idx, steps[i] : steps[i + 1], :, :]
                seed = np.unravel_index(
                    np.argmax(heatmap_section), heatmap_section.shape
                )  # TODO transpose here
                seed = np.array(seed) + np.array([steps[i], 0, 0])
                seeds_dict[label_name][i + 1] = seed

            # Last seed point on max_slice
            seed = np.unravel_index(
                np.argmax(heatmap_np[label_idx, max_slice]), slice_shape
            )  # TODO transpose here
            seeds_dict[label_name][-1] = np.array([max_slice, *seed])

        return seeds_dict

    @staticmethod
    def get_centerline(
        seeds_dict: Dict[str, np.ndarray], heatmap_np: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Connect seeds with Dijkstra algorithm to obtain one center per slice.

        Args:
            seeds_dict: dictionary with keys corresponding to labels (internal, external) and items to the array of seeds.
            heatmap_np: heatmap from which the seeds are extracted.

        Returns:
            dictionary with keys corresponding to labels (internal, external) and items to the array of centers.
        """
        # cost based on heatmap only
        cost_np = np.max(heatmap_np) - heatmap_np

        paths = {"internal": [], "external": []}
        for label_idx, label_name in enumerate(["internal", "external"]):
            label_cost_np = cost_np[label_idx]
            seeds_np = seeds_dict[label_name]
            for seed_idx in range(seeds_np.shape[0] - 1):
                # Restrict possible space between seeds only
                seed_cost_np = np.copy(label_cost_np)
                seed_cost_np[: int(seeds_np[seed_idx, 0])] = np.inf
                seed_cost_np[int(seeds_np[seed_idx + 1, 0]) + 1 :] = np.inf

                paths[label_name].append(
                    dijkstra(
                        seed_cost_np,
                        seeds_np[seed_idx],
                        seeds_np[seed_idx + 1],
                    )
                )
            paths[label_name] = np.vstack(paths[label_name])

            # resample so we have one center per axial slice
            interp = interp1d(
                paths[label_name][:, 0], paths[label_name][:, [1, 2]].transpose()
            )  # TODO transpose

            # Interpolate between min_slice and max_slice
            slices = np.arange(seeds_np[0, 0], seeds_np[-1, 0])
            paths[label_name] = np.concatenate(
                [np.expand_dims(slices, 1), interp(slices.T).T], axis=1
            )
        return paths
