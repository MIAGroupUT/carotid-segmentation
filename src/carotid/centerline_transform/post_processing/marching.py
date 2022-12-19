import numpy as np
from typing import Dict, Any, Optional
import warnings
from dijkstra3d import dijkstra
from scipy.interpolate import interp1d
from .template import CenterlineExtractor

side_list = ["left", "right"]


class MarchingExtractor(CenterlineExtractor):
    def __init__(
        self,
        max_intensity_threshold: float,
        step_size: int,
        spatial_threshold: int,
        cost_rule: str,
        cut_common: bool = True,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.max_intensity_threshold = max_intensity_threshold
        self.step_size = step_size
        self.spatial_threshold = spatial_threshold
        self.cost_rule = cost_rule
        self.cut_common = cut_common
        self.epsilon = epsilon

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:

        return {
            side: self.process_image_heatmap_pair(
                sample["img"], sample[f"{side}_label"]
            )
            for side in side_list
        }

    def process_image_heatmap_pair(
        self,
        image_np: np.ndarray,
        heatmap_np: np.ndarray,
    ) -> Dict[str, np.ndarray]:

        intensity_threshold_list = np.array([0.1, 0.1])
        empty_array = np.zeros((0, 5))

        # Find root
        root_coords = self.find_root(heatmap_np)

        # Initialize containers for both labels
        label_coords = [root_coords, root_coords]
        total_label_path_list = [empty_array, empty_array]
        slice_idx_list = [None, None]
        latest_valid_slice = [
            None,
            None,
        ]  # Last slice on which the path is considered as valid.
        # Default final decision at the end of the loop will be to increase the thresholds
        # for both thresholds
        increase_threshold = [True, True]

        # Normalize image and heatmap
        heatmap_np = np.clip(heatmap_np, a_min=0, a_max=None)
        heatmap_np /= (np.max(heatmap_np, axis=(2, 3)) + self.epsilon)[
            :, :, np.newaxis, np.newaxis
        ]

        total_common_np = self.compute_lowest_slices_path(
            image_np, heatmap_np, root_coords
        )
        # Remove parts with a high uncertainty
        if self.cut_common:
            to_remove = total_common_np[:, 3] > self.max_intensity_threshold
            total_common_np = total_common_np[~to_remove]

        while (intensity_threshold_list <= self.max_intensity_threshold).any():
            # Go up through the internal and external carotids
            for label_idx in range(2):
                intensity_threshold = intensity_threshold_list[label_idx]
                # If compute additional path only for paths with low thresholds
                if intensity_threshold <= self.max_intensity_threshold:
                    label_np = self.compute_highest_paths(
                        image_np,
                        heatmap_np,
                        label_idx,
                        label_coords[label_idx],
                        slice_idx=slice_idx_list[label_idx],
                        intensity_threshold=intensity_threshold,
                    )

                    if len(label_np) > 0:
                        total_label_path_list[label_idx] = self.concatenate_and_replace(
                            total_label_path_list[label_idx], label_np
                        )
                        # Set new coords to start from
                        label_coords[label_idx] = label_np[-1][:3:].astype(int)
                        slice_idx_list[label_idx] = None

            # If at least one path could not be drawn, increase all thresholds
            if len(total_label_path_list[0]) == 0 or len(total_label_path_list[1]) == 0:
                increase_threshold = [True, True]
            # Else look for different cases
            else:
                label_coords_dist = np.linalg.norm(label_coords[0] - label_coords[1])

                # If both labels end at the same location (still common path)
                if label_coords_dist < self.spatial_threshold:
                    increase_threshold = [True, True]

                else:
                    for label_idx in range(2):
                        comparison_idx = (label_idx + 1) % 2
                        # Dist between end of current label and any point on comparison path
                        dist = np.linalg.norm(
                            total_label_path_list[comparison_idx][:, :3]
                            - label_coords[label_idx],
                            axis=1,
                        )
                        # Current label ends somewhere on the other path
                        if (dist < self.spatial_threshold).any():
                            # Do not increase threshold except same decision was taken before
                            increase_threshold[label_idx] = not increase_threshold[
                                label_idx
                            ]
                            # Next time start to look for seed further
                            slice_idx_list[label_idx] = (
                                label_coords[label_idx][0] - self.step_size
                            )
                            # Next time start drawing from root again
                            label_coords[label_idx] = root_coords
                            # Forget what we stored before until last valid slice
                            if latest_valid_slice[label_idx] is None:
                                total_label_path_list[label_idx] = empty_array
                            else:
                                total_label_path_list[
                                    label_idx
                                ] = self.remove_above_slice(
                                    total_label_path_list[label_idx],
                                    latest_valid_slice[label_idx],
                                )
                        else:
                            increase_threshold[label_idx] = True
                            latest_valid_slice[label_idx] = label_coords[label_idx][0]

            # Apply decision on thresholds
            for label_idx, decision in enumerate(increase_threshold):
                if decision:
                    intensity_threshold_list[label_idx] += 0.1

        return {
            "internal": self.resample_path(
                np.concatenate([total_label_path_list[0][::-1], total_common_np])
            ),
            "external": self.resample_path(
                np.concatenate([total_label_path_list[1][::-1], total_common_np])
            ),
        }

    @staticmethod
    def concatenate_uncertainty(
        path_np: np.ndarray,
        image_np: np.ndarray,
        heatmap_np: np.ndarray,
        label_idx: int = None,
        starting_image_intensity: float = 0,
    ) -> np.ndarray:

        # Add two columns to path (image and heatmap uncertainties).
        uncertainty_np = np.append(path_np, np.zeros((len(path_np), 2)), axis=1)

        if label_idx is None:
            heatmap_np = np.mean(heatmap_np, axis=0)[np.newaxis, :]
            label_idx = 0

        max_image_intensity = starting_image_intensity
        for point_idx, point in enumerate(path_np):
            image_value = image_np[0][tuple(point)]
            heatmap_value = heatmap_np[label_idx][tuple(point)]
            if image_value > max_image_intensity:
                max_image_intensity = image_value
            uncertainty_np[point_idx, -2] = max_image_intensity
            uncertainty_np[point_idx, -1] = 1 - heatmap_value

        return uncertainty_np

    def find_root(
        self,
        heatmap_np: np.ndarray,
    ) -> np.ndarray:
        """Find root seed in the common carotid artery to start centerline tracking"""
        sum_np = np.sum(heatmap_np, axis=0)

        initial_coords = np.unravel_index(np.argmax(sum_np), sum_np.shape)
        initial_slice, initial_x, initial_y = initial_coords

        for label_idx in range(2):
            label_np = heatmap_np[label_idx, initial_slice]
            label_seed = np.unravel_index(np.argmax(label_np), label_np.shape)
            label_x, label_y = label_seed

            if (
                np.linalg.norm(np.array((initial_x - label_x, initial_y - label_y)))
                > self.spatial_threshold
            ):
                warnings.warn("The starting seed found may not be reliable.")

        return np.array(initial_coords)

    @staticmethod
    def check_image_heatmap_normalized(image_np: np.ndarray, heatmap_np: np.ndarray):
        assert np.max(heatmap_np) <= 1
        assert np.max(image_np) <= 1
        assert np.min(heatmap_np) >= 0
        assert np.min(image_np) >= 0

        n_labels, _, _, _ = heatmap_np.shape
        assert n_labels == 2

    def compute_lowest_slices_path(
        self,
        image_np: np.ndarray,
        heatmap_np: np.ndarray,
        root_coords: np.ndarray,
    ) -> np.ndarray:
        """
        From a point found in the common carotid, go down in the common carotid artery as far as possible.
        The heatmap and images are assumed to be already normalized between 0 and 1.
        The process stops as soon as the internal and external heatmaps disagree on the best seed.

        Args:
            image_np: Array of the MRI.
            heatmap_np: Array of the heatmap segmentation of the internal and external carotids.
            root_coords: (Z, X, Y) coordinates of the root seed lying in the common carotid artery.

        Returns:
            Coordinates of points forming the common carotid artery.

        """
        self.check_image_heatmap_normalized(image_np, heatmap_np)

        # Initialization
        _, max_z, _, _ = heatmap_np.shape
        root_slice, _, _ = root_coords
        slice_idx = root_slice + self.step_size
        previous_coords = root_coords
        current_uncertainty = 0
        resume = True

        # Compute once everything needed for the cost function of Dijkstra algorithm
        inv_image_np = np.max(image_np) - image_np
        prod_np = heatmap_np * inv_image_np
        if self.cost_rule == "mix":
            cost_np = np.mean(1 - prod_np, axis=0)  # Mean value of both labels
        else:
            cost_np = image_np[0].numpy().copy()

        cost_np[root_slice:] = np.max(
            cost_np
        )  # Prevents from looking above the root slice

        # Results container 3 coordinates + 2 uncertainty columns
        total_path_np = np.zeros((0, 5))

        while slice_idx < max_z and resume:
            # Look for seed with maximum intensities in each label
            seed_list = [None, None]
            for label_idx in range(2):
                prod_slice_np = prod_np[label_idx, slice_idx]
                seed_list[label_idx] = np.array(
                    np.unravel_index(np.argmax(prod_slice_np), prod_slice_np.shape)
                )

            seeds_distance = np.linalg.norm(np.diff(seed_list, axis=0))

            # If external and internal maps disagree on the seed, stop the process.
            if seeds_distance > self.spatial_threshold:
                resume = False
            else:
                new_seed = seed_list[0]

                slice_cost_np = cost_np.copy()
                # Prevents from looking below current slice
                slice_cost_np[slice_idx + 1 : :] = cost_np.max()

                new_coords = np.insert(new_seed, 0, slice_idx)
                path_np = dijkstra(
                    slice_cost_np,
                    previous_coords,
                    new_coords,
                )

                path_np = self.concatenate_uncertainty(
                    path_np,
                    image_np,
                    heatmap_np,
                    starting_image_intensity=current_uncertainty,
                )
                total_path_np = self.concatenate_and_replace(total_path_np, path_np)
                previous_coords = new_coords
                current_uncertainty = np.max(total_path_np[:, 3])

            slice_idx += self.step_size

        return total_path_np

    def compute_highest_paths(
        self,
        image_np: np.ndarray,
        heatmap_np: np.ndarray,
        label_idx: int,
        root_coords: np.ndarray,
        intensity_threshold: float,
        slice_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        From a point found in the common carotid, go up in the internal and external carotid arteries as far as possible.
        The heatmap and images are assumed to be already normalized between 0 and 1.

        Args:
            image_np: Array of the MRI.
            heatmap_np: Array of the heatmap segmentation of the internal and external carotids.
            label_idx: which label is being tracked {0: internal, 1: external}.
            root_coords: (Z, Y, X) coordinates of the root seed lying in the common carotid artery.
            intensity_threshold: limit of uncertainty authorized to keep a seed. Increasing the value of this parameter
                may produce longer centerlines, but less reliable.
            slice_idx: starting slice to find new seeds of the centerline.
               If None will be taken at root_Z - step_size

        Returns:
            Coordinates of points constituting the internal and external carotids.

        """
        self.check_image_heatmap_normalized(image_np, heatmap_np)

        # Initialization
        root_slice, _, _ = root_coords
        previous_coords = root_coords
        if slice_idx is None:
            slice_idx = root_slice - self.step_size
        current_uncertainty = 0
        total_path_np = np.zeros((0, 5))

        # Compute once everything needed for the cost function of Dijkstra algorithm
        inv_image_np = np.max(image_np) - image_np
        prod_np = heatmap_np * inv_image_np
        if self.cost_rule == "mix":
            cost_np = 1 - prod_np
            cost_np = cost_np[label_idx]
        else:
            cost_np = image_np[0].numpy()

        cost_np[root_slice::] = 1
        slice_shape = image_np[0, 0].shape

        while slice_idx > 0:

            tmp_cost_np = cost_np.copy()
            tmp_cost_np[:slice_idx:] = 1

            seed = np.unravel_index(
                np.argmax(prod_np[label_idx, slice_idx]), slice_shape
            )

            # Look for link with root seed
            new_coords = np.insert(seed, 0, slice_idx)
            path_np = dijkstra(
                tmp_cost_np,
                previous_coords,
                new_coords,
            )

            path_np = self.concatenate_uncertainty(
                path_np=path_np,
                image_np=image_np,
                heatmap_np=heatmap_np,
                label_idx=label_idx,
                starting_image_intensity=current_uncertainty,
            )
            current_uncertainty = np.max(path_np[:, 3])

            if current_uncertainty < intensity_threshold:
                previous_coords = new_coords
                total_path_np = np.concatenate([total_path_np, path_np])

            slice_idx -= self.step_size

        return total_path_np

    @staticmethod
    def resample_path(point_cloud: np.ndarray) -> np.ndarray:
        """Resample array according to the first column."""
        interpolator = interp1d(point_cloud[:, 0], point_cloud[:, 1::].transpose())
        slices = np.arange(
            np.ceil(np.min(point_cloud[:, 0])), np.floor(np.max(point_cloud[:, 0])) + 1
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            interpolated_pc = interpolator(slices).transpose()
        point_cloud = np.concatenate(
            [np.expand_dims(slices, 1), interpolated_pc], axis=1
        )
        nan_rows = np.any(np.isnan(point_cloud), axis=1)
        point_cloud = point_cloud[~nan_rows]
        return point_cloud

    @staticmethod
    def concatenate_and_replace(
        total_path: np.ndarray, addition_path: np.ndarray
    ) -> np.ndarray:
        """
        Remove slices already present in total_path and concatenate addition_path
        """
        min_slice = addition_path[:, 0].min()
        max_slice = addition_path[:, 0].max()

        to_remove = (total_path[:, 0] >= min_slice) & (total_path[:, 0] <= max_slice)
        total_path = total_path[~to_remove]
        return np.concatenate([total_path, addition_path])

    @staticmethod
    def remove_above_slice(total_path_np: np.ndarray, slice_idx: int):
        to_remove = total_path_np[:, 0] > slice_idx
        return total_path_np[~to_remove]
