from typing import List, Dict, Set
from os import path, makedirs, listdir
import numpy as np
import torch
from monai.transforms import Transform, MapTransform
from monai.config import KeysCollection
from .template import Serializer
from .errors import MissingProcessedObjException
from ..logger import write_json, read_json

side_list = ["left", "right"]


class LoadPolarDird(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            dir_path = d[key]
            d[key] = list()
            file_list = [
                filename
                for filename in listdir(dir_path)
                if filename.endswith("_polar.npy")
            ]
            file_list.sort()
            for filename in file_list:
                try:
                    polar_path = path.join(dir_path, filename)
                    polar_np = np.load(polar_path, allow_pickle=True).astype(float)
                    center_path = polar_path[:-10] + "_center.npy"
                    center_np = np.load(center_path, allow_pickle=True).astype(float)
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
                        "polar_pt": torch.from_numpy(polar_np).float().unsqueeze(0),
                        "center": center_np,
                    },
                )

            # Read meta dict
            json_path = path.join(dir_path, "parameters.json")
            d[f"{key}_meta_dict"] = read_json(json_path)

        return d


class PolarSerializer(Serializer):
    """
    Read and write outputs of polar_transform.
    For each side this corresponds to a list of dictionaries with the following keys:
        - "label": str is the label "external" or "internal",
        - "slice_idx": int is the index of the axial slice,
        - "polar_pt": Tensor is the tensor wrapping the 3D polar image,
        - "center": array is the array of the 3 spatial coordinates of the center.
    Each side is also associated to a meta dict including the parameters of the polar transform.
    """

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
            polar_dir = path.join(
                self.parameters["dir"], sample["participant_id"], "polar_transform"
            )
            sample["left_polar"] = path.join(polar_dir, "left_polar")
            sample["right_polar"] = path.join(polar_dir, "right_polar")

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
                polar_np = polar_dict["polar_pt"].squeeze(0).numpy()
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
