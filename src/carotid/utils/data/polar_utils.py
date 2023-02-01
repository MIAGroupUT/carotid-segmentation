from typing import List, Dict, Set
from os import path, makedirs, listdir
import numpy as np
from monai.transforms import Transform
from .template import Serializer
from carotid.utils.transforms import LoadPolarDird

side_list = ["left", "right"]


class PolarSerializer(Serializer):
    def get_transforms(self) -> List[Transform]:
        return [LoadPolarDird(keys=["left_polar", "right_polar"])]

    def find_participant_set(self) -> Set[str]:
        return {
            participant_id
            for participant_id in listdir(self.parameters["dir"])
            if path.exists(
                path.join(self.parameters["dir"], participant_id, "polar_transform")
            )
        }

    def add_path(self, sample_list: List[Dict[str, str]]):
        for sample in sample_list:
            heatmap_dir = path.join(
                self.parameters["dir"], sample["participant_id"], "polar_transform"
            )
            sample["left_polar"] = path.join(heatmap_dir, "left_polar")
            sample["right_polar"] = path.join(heatmap_dir, "right_polar")

    def write(self, sample):
        output_dir = path.join(
            self.parameters["dir"], sample["participant_id"], "polar_transform"
        )
        makedirs(output_dir, exist_ok=True)

        for side in side_list:
            side_dir = path.join(output_dir, f"{side}_polar")
            makedirs(side_dir, exist_ok=True)
            polar_list = sample[f"{side}_polar"]
            for polar_dict in polar_list:
                label_name = polar_dict["label"]
                slice_idx = polar_dict["slice_idx"]
                polar_np = polar_dict["polar_img"]
                np.save(
                    path.join(
                        side_dir, f"label-{label_name}_slice-{slice_idx}_polar.npy"
                    ),
                    polar_np,
                )
