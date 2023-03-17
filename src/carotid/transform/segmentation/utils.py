import warnings
from shapely import Polygon
from rasterio.features import rasterize
from typing import Dict, Any
import numpy as np
from monai.data.meta_tensor import MetaTensor


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
            affine = sample[f"{side}_contour_meta_dict"]["affine"]
            # need to transpose for rasterize assuming we are working with images and not tensors
            slice_shape = (orig_shape[1], orig_shape[0])

            # Create new segmentation array
            segmentation_np = np.zeros((2, 2, *orig_shape))

            for object_idx, object_name in enumerate(self.object_list):
                object_df = contour_df[contour_df.object == object_name]
                for label_idx, label_name in enumerate(self.label_list):
                    label_cloud = object_df[object_df.label == label_name][
                        ["x", "y", "z"]
                    ].values
                    if len(label_cloud) == 0:
                        warnings.warn(
                            f"No point cloud exists for the {side} {label_name} carotid."
                        )
                    else:
                        min_slice = int(min(label_cloud[:, 2]))
                        max_slice = int(max(label_cloud[:, 2]))

                        # For each slice find a convex hull enclosing the contour and fill it
                        for slice_idx in range(min_slice, max_slice + 1):
                            slice_cloud = label_cloud[label_cloud[:, 2] == slice_idx][
                                :, [0, 1]
                            ]
                            if len(slice_cloud) > 0:
                                poly = Polygon(slice_cloud)
                                hull = poly.convex_hull
                                slice_img = rasterize(
                                    [hull], out_shape=slice_shape, fill=0, default_value=1
                                ).T
                                segmentation_np[label_idx, object_idx, :, :, slice_idx] += slice_img
            segmentation_np = segmentation_np.reshape((4, *orig_shape))
            sample[f"{side}_segmentation"] = MetaTensor(segmentation_np, affine=affine)

        return sample
