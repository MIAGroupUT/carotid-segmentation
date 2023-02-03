from typing import List, Dict, Any, Set
from monai.transforms import Transform, LoadImaged
from os import path, listdir, makedirs
import numpy as np
from .template import Serializer


class SegmentationSerializer(Serializer):
    """
    Read and write outputs of segmentation_transform.
    For each side this corresponds to a point cloud of size (2, M, 3).
    Description of each axis:
        - index 0 is the lumen, index 1 is the wall,
        - M is the number of points in the cloud,
        - spatial coords, ordered as Z, Y, X.
    """

    def get_transforms(self) -> List[Transform]:
        transforms = [
            LoadImaged(
                keys=["left_segmentation", "right_segmentation"], reader="numpyreader"
            )
        ]
        return transforms

    def find_participant_set(self) -> Set[str]:
        return {
            participant_id
            for participant_id in listdir(self.parameters["dir"])
            if path.exists(
                path.join(
                    self.parameters["dir"], participant_id, "segmentation_transform"
                )
            )
        }

    def add_path(self, sample_list: List[Dict[str, str]]):
        for sample in sample_list:
            segmentation_dir = path.join(
                self.parameters["dir"],
                sample["participant_id"],
                "segmentation_transform",
            )
            sample["left_segmentation"] = path.join(
                segmentation_dir, "left_segmentation.npy"
            )
            sample["right_segmentation"] = path.join(
                segmentation_dir, "right_segmentation.npy"
            )

    def write(self, sample: Dict[str, Any]):
        side_list = ["left", "right"]
        output_dir = path.join(
            self.parameters["dir"], sample["participant_id"], "segmentation_transform"
        )
        makedirs(output_dir, exist_ok=True)

        for side in side_list:
            segmentation_np = sample[f"{side}_segmentation"]
            np.save(path.join(output_dir, f"{side}_segmentation.npy"), segmentation_np)
