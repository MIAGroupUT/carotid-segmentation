import abc
import numpy as np
import torch
from dijkstra3d import dijkstra
from scipy.interpolate import interp1d
from typing import Dict, Any, Tuple
import pandas as pd


side_list = ["left", "right"]


class CenterlineExtractor:
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    @abc.abstractmethod
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
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

    def get_seedpoints(self, heatmap_pt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Find seeds that will be connected by the Dijkstra algorithm.

        Args:
            heatmap_pt: heatmap from which the seeds are extracted.

        Returns:
            dictionary with keys corresponding to labels (internal, external) and items to the array of seeds.
        """
        # TODO take into image transpose
        slice_shape = heatmap_pt[0, 0].shape
        mask_pt = heatmap_pt > self.parameters["threshold"]

        # only determine seed points where both internal/external carotids exceed threshold
        (valid_slice_indices,) = torch.where(mask_pt.any(-1).any(-1).all(0))
        min_slice = valid_slice_indices[0].item()
        max_slice = valid_slice_indices[-1].item()
        steps = torch.arange(min_slice, max_slice, self.parameters["step_size"])

        seeds_dict = {
            "internal": torch.zeros((len(steps) + 1, 3)),
            "external": torch.zeros((len(steps) + 1, 3)),
        }

        for label_idx, label_name in enumerate(["internal", "external"]):
            # First seed on min_slice
            seed = unravel_index(
                torch.argmax(heatmap_pt[label_idx, min_slice]),
                slice_shape,
            )  # TODO transpose here
            seeds_dict[label_name][0] = torch.Tensor([min_slice, *seed])

            # Seeds in each 3D section
            for i in range(len(steps) - 1):
                heatmap_section = heatmap_pt[label_idx, steps[i] : steps[i + 1], :, :]
                seed = unravel_index(
                    torch.argmax(heatmap_section), heatmap_section.shape
                )  # TODO transpose here
                seed = torch.Tensor(seed) + torch.Tensor([steps[i], 0, 0])
                seeds_dict[label_name][i + 1] = seed

            # Last seed point on max_slice
            seed = unravel_index(
                torch.argmax(heatmap_pt[label_idx, max_slice]), slice_shape
            )  # TODO transpose here
            seeds_dict[label_name][-1] = torch.Tensor([max_slice, *seed])

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


def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.
    Source: https://github.com/pytorch/pytorch/issues/35674

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode="floor")

    coord = torch.stack(coord[::-1], dim=-1).long()

    return coord


def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> Tuple[torch.LongTensor, ...]:
    r"""Converts flat indices into unraveled coordinates in a target shape.
    Source: https://github.com/pytorch/pytorch/issues/35674

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (N,).
        shape: The targeted shape, (D,).

    Returns:
        A tuple of unraveled coordinate tensors of shape (D,).
    """

    coord = unravel_indices(indices, shape)
    return tuple(coord)
