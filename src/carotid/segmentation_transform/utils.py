import warnings
from shapely import Polygon
from rasterio.features import rasterize
from typing import Dict, Any
import numpy as np


class SegmentationTransform:
    """
    Transform DataFrame containing the point clouds of the lumen and wall of internal and external carotids
    in a voxel mask.
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        self.parameters = parameters
        self.side_list = ["left", "right"]
        self.label_list = ["internal", "external"]
        self.object_list = ["lumen", "wall"]

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        for side in self.side_list:
            # Retrieve contour data
            contour_df = sample[f"{side}_contour"]
            orig_shape = sample[f"{side}_contour_meta_dict"]["orig_shape"]
            # need to transpose for rasterize assuming we are working with images and not tensors
            slice_shape = (orig_shape[2], orig_shape[1])

            # Create new segmentation entry
            sample[f"{side}_segmentation"] = dict()

            for object_name in self.object_list:
                object_df = contour_df[contour_df.object == object_name]
                segmentation_np = np.zeros(orig_shape)
                for label_name in self.label_list:
                    label_cloud = object_df[object_df.label == label_name][
                        ["z", "y", "x"]
                    ].values
                    if len(label_cloud) == 0:
                        warnings.warn(
                            f"No point cloud exists for the {side} {label_name} carotid."
                        )
                    else:
                        min_slice = int(min(label_cloud[:, 0]))
                        max_slice = int(max(label_cloud[:, 0]))

                        # For each slice find a convex hull enclosing the contour and fill it
                        for slice_idx in range(min_slice, max_slice + 1):
                            slice_cloud = label_cloud[label_cloud[:, 0] == slice_idx][
                                :, [1, 2]
                            ]
                            poly = Polygon(slice_cloud)
                            hull = poly.convex_hull
                            slice_img = rasterize(
                                [hull], out_shape=slice_shape, fill=0, default_value=1
                            ).T
                            segmentation_np[slice_idx] += slice_img
                segmentation_np = np.clip(segmentation_np, a_min=0, a_max=1)
                sample[f"{side}_{object_name}_segmentation"] = segmentation_np

            # Remove lumen segmentation from wall segmentation
            sample[f"{side}_wall_segmentation"] = (
                sample[f"{side}_wall_segmentation"]
                - sample[f"{side}_lumen_segmentation"]
            )

        return sample
