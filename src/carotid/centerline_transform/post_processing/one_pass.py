import torch
import numpy as np
from typing import Dict, Any, List
from dijkstra3d import dijkstra
from scipy.interpolate import interp1d
from template import CenterlineExtractor

side_list = ["left", "right"]


class OnePassExtractor(CenterlineExtractor):
    """
    Extracts centerline from heatmaps by seeding them once.
    """

    def __init__(
        self,
        step_size: int = 25,
        threshold: float = 75,
        **kwargs,
    ):
        super().__init__()
        self.step_size = step_size
        self.threshold = threshold

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:

        seedpoints = {
            side: self.get_seedpoints(sample[f"{side}_label"]) for side in side_list
        }
        paths = {
            side: self.get_centerline(seedpoints[side], sample[f"{side}_label"])
            for side in side_list
        }

        return paths

    def get_seedpoints(self, heatmap: np.ndarray) -> Dict[str, np.ndarray]:
        # obtain seedpoints for the Dijkstra algorithm, for both internal and external carotid artery, based on the predicted heatmap
        # return numpy arrays, as dijkstra algorithm takes numpy as input
        seeds = {"internal": [], "external": []}
        # cut based on heatmaps of both external / internal carotid
        masked_heatmap = (torch.from_numpy(heatmap > self.threshold)).int()
        # only determine seed points where both internal/external carotids exceed threshold
        min_slice = max(
            torch.min(torch.nonzero(masked_heatmap[0, :, :, :])[:, 0]).item(),
            torch.min(torch.nonzero(masked_heatmap[1, :, :, :])[:, 0]).item(),
        )
        max_slice = min(
            torch.max(torch.nonzero(masked_heatmap[0, :, :, :])[:, 0]).item(),
            torch.max(torch.nonzero(masked_heatmap[1, :, :, :])[:, 0]).item(),
        )
        steps = np.append(np.arange(min_slice, max_slice, self.step_size), [max_slice])
        for index, carotid in enumerate(["internal", "external"]):
            # first seedpoint on min_slice
            first_seed = self.unravel_index(
                torch.argmax(heatmap[index, steps[0], :, :]),
                heatmap[index, steps[0], :, :].shape,
            )
            seeds[carotid].append(np.array([min_slice, first_seed[0], first_seed[1]]))
            for i in range(len(steps) - 1):
                heatmap_section = heatmap[index, steps[i] : steps[i + 1], :, :]
                seed = self.unravel_index(
                    torch.argmax(heatmap_section), heatmap_section.shape
                )
                seeds[carotid].append(np.array(seed) + np.array([steps[i], 0, 0]))
            # add last seedpoint on max_slice
            last_seed = self.unravel_index(
                torch.argmax(heatmap[index, steps[-1], :, :]),
                heatmap[index, steps[-1], :, :].shape,
            )
            seeds[carotid].append(np.array([max_slice, last_seed[0], last_seed[1]]))
            seeds[carotid] = np.unique(np.asarray(seeds[carotid]), axis=0)
        return seeds

    @staticmethod
    def get_centerline(
        seedpoints: Dict[str, np.ndarray], cost_map: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # in case of based on heatmap
        cost_map = -1 * cost_map + np.ones_like(cost_map) * np.max(cost_map)
        paths = {"internal": [], "external": []}
        for index, carotid in enumerate(["internal", "external"]):
            for j in range(seedpoints[carotid].shape[0] - 1):
                paths[carotid].append(
                    dijkstra(
                        cost_map[index, :, :, :],
                        seedpoints[carotid][j, :],
                        seedpoints[carotid][j + 1, :],
                    )
                )
            paths[carotid] = np.vstack(paths[carotid])
            # resample so we have one center per axial slice!
            interp = interp1d(
                paths[carotid][:, 0], paths[carotid][:, [1, 2]].transpose()
            )
            slices = np.arange(
                np.maximum(3, seedpoints[carotid][0, 0]),
                np.minimum(seedpoints[carotid][-1, 0], 284),
            )  # /!\ 284 based on afterfifteen data!
            paths[carotid] = np.concatenate(
                [np.expand_dims(slices, 1), interp(slices.T).T], axis=1
            )
        return paths

    @staticmethod
    def unravel_index(index: torch.Tensor, shape: torch.Size) -> List[int]:
        # torch version of the function np.unravel_index
        out = []
        for dim in reversed(shape):
            out.append((index % dim).item())
            index = torch.div(index, dim, rounding_mode="floor")
        return list(reversed(out))
