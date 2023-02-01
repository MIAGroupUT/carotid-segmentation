from typing import List, Dict, Set
from os import path, makedirs, listdir
import numpy as np
from monai.transforms import Transform, MapTransform
from monai.config import KeysCollection
from .template import Serializer
from .errors import MissingProcessedObjException
from ..logger import write_json

side_list = ["left", "right"]


class LoadPolarDird(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            polar_path = d[key]
            d[key] = list()
            file_list = [
                filename
                for filename in listdir(polar_path)
                if filename.endswith("_center.npy")
            ]
            file_list.sort()
            for filename in file_list:
                try:
                    polar_np = np.load(
                        path.join(polar_path, filename), allow_pickle=True
                    ).astype(float)
                except FileNotFoundError:
                    raise MissingProcessedObjException(
                        f"Polar image at path {path.join(polar_path, filename)} was not found.\n"
                        f"Please ensure that the polar_transform was run in your experiment folder."
                    )
                label_name = filename.split("_")[0].split("-")[1]
                slice_idx = filename.split("_")[1].split("-")[1]
                d[key].append(
                    {
                        "label": label_name,
                        "slice_idx": slice_idx,
                        "polar_img": polar_np,
                    },
                )
        return d


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
                center_np = polar_dict["center"]
                np.save(
                    path.join(
                        side_dir, f"label-{label_name}_slice-{slice_idx}_polar.npy"
                    ),
                    polar_np,
                )
                np.save(
                    path.join(
                        side_dir, f"label-{label_name}_slice-{slice_idx}_center.npy"
                    ),
                    center_np,
                )
            write_json(
                sample[f"{side}_polar_meta_dict"],
                path.join(side_dir, "parameters.json"),
            )
