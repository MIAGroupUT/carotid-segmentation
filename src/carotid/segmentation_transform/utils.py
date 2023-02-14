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
            contour_df = sample[f"{side}_contour"]
            for object_name in self.object_list:
                object_df = contour_df[contour_df.object == object_name]
                segmentation_np = np.zeros()
                for label_name in self.label_list:
                    point_cloud = object_df[object_df.label == label_name][
                        ["z", "y", "x"]
                    ].values
