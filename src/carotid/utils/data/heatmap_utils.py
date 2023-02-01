from monai.transforms import (
    LoadImaged,
    Transform,
)
from .template import Serializer
from os import path, listdir, makedirs
import numpy as np
from typing import List, Dict, Any, Set


class HeatmapSerializer(Serializer):
    def get_transforms(self) -> List[Transform]:
        transforms = [
            LoadImaged(keys=["left_heatmap", "right_heatmap"], reader="numpyreader")
        ]
        return transforms

    def find_participant_set(self) -> Set[str]:
        return {
            participant_id
            for participant_id in listdir(self.parameters["dir"])
            if path.exists(
                path.join(self.parameters["dir"], participant_id, "heatmap_transform")
            )
        }

    def add_path(self, sample_list: List[Dict[str, str]]):
        for sample in sample_list:
            heatmap_dir = path.join(
                self.parameters["dir"], sample["participant_id"], "heatmap_transform"
            )
            sample["left_heatmap"] = path.join(heatmap_dir, "left_heatmap.npy")
            sample["right_heatmap"] = path.join(heatmap_dir, "right_heatmap.npy")

    def write(self, sample: Dict[str, Any]):
        side_list = ["left", "right"]
        output_dir = path.join(
            self.parameters["dir"], sample["participant_id"], "heatmap_transform"
        )
        makedirs(output_dir, exist_ok=True)

        for side in side_list:
            heatmap_np = sample[f"{side}_heatmap"]
            np.save(path.join(output_dir, f"{side}_heatmap.npy"), heatmap_np)
            pass
