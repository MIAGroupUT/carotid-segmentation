import json
import toml
import abc
from os import path, listdir, makedirs
import numpy as np
import torch
import pandas as pd
from monai.transforms import (
    MapTransform,
    LoadImaged,
    ScaleIntensityRangePercentilesd,
    Orientationd,
    Compose,
)
from monai.config import KeysCollection
from typing import Dict, Any, Union, Optional, List, Set
from .errors import MissingProcessedObjException, MissingRawArgException


def write_json(parameters: Dict[str, Any], json_path: str) -> None:
    """Writes parameters dictionary at json_path."""
    json_data = json.dumps(parameters, skipkeys=True, indent=4)
    with open(json_path, "w") as f:
        f.write(json_data)


def read_json(json_path: str) -> Dict[str, Any]:
    """Reads JSON file at json_path and returns corresponding dictionary."""
    with open(json_path, "r") as f:
        parameters = json.load(f)

    return parameters


def read_and_fill_default_toml(
    config_path: Optional[Union[str, Dict[str, Any]]], default_path: str
) -> Dict[str, Any]:
    default_parameters = toml.load(default_path)

    if config_path is None:
        return default_parameters
    elif isinstance(config_path, str):
        config_parameters = toml.load(config_path)
    elif isinstance(config_path, dict):
        config_parameters = config_path.copy()
    else:
        raise ValueError(
            f"First argument should be a path to a TOML file or a dictionary of parameters."
            f"Current value is {config_path}."
        )

    for param, value in default_parameters.items():
        if param not in config_parameters:
            config_parameters[param] = value

    return config_parameters


##############
# TRANSFORMS #
##############
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
                polar_path = path.join(dir_path, filename)
                try:
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
                        "center_pt": torch.from_numpy(center_np).float(),
                    },
                )

        return d


class LoadCSVd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False, sep=","):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.sep = sep

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            try:
                d[key] = pd.read_csv(d[key], sep=self.sep)
            except FileNotFoundError:
                MissingProcessedObjException(
                    f"TSV file {key} was not found.\n"
                    f"Please ensure that the centerline_transform was run in your experiment folder."
                )

        return d


class LoadMetaJSONd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        json_name: str = "parameters.json",
        parent_dir: bool = True,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.json_name = json_name
        self.parent_dir = parent_dir

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.parent_dir:
                dir_path = path.dirname(d[key])
            else:
                dir_path = d[key]
            try:
                d[f"{key}_meta_dict"] = read_json(path.join(dir_path, self.json_name))
            except FileNotFoundError:
                raise MissingProcessedObjException(
                    f"TSV file {key} was not found.\n"
                    f"Please ensure that the centerline_transform was run in your experiment folder."
                )

        return d


class Serializer:
    def __init__(
        self,
        dir_path: str,
        transform_name: str,
        file_ext: Optional[str],
        monai_reader: Union[MapTransform, Compose],
    ):
        self.dir_path = dir_path
        self.transform_name = transform_name
        if file_ext is None:
            self.file_end = ""
        else:
            self.file_end = f".{file_ext}"
        self.monai_reader = monai_reader
        self.side_list = ["left", "right"]

    def find_participant_set(self) -> Set[str]:
        return {
            participant_id
            for participant_id in listdir(self.dir_path)
            if path.exists(
                path.join(
                    self.dir_path, participant_id, f"{self.transform_name}_transform"
                )
            )
        }

    def add_path(self, sample_list: List[Dict[str, str]]):
        for sample in sample_list:
            output_dir = path.join(
                self.dir_path,
                sample["participant_id"],
                f"{self.transform_name}_transform",
            )
            for side in self.side_list:
                sample[f"{side}_{self.transform_name}"] = path.join(
                    output_dir, f"{side}_{self.transform_name}{self.file_end}"
                )

    def write(self, sample: Dict[str, Any]):
        output_dir = path.join(
            self.dir_path, sample["participant_id"], f"{self.transform_name}_transform"
        )
        makedirs(output_dir, exist_ok=True)

        for side in self.side_list:
            output_path = path.join(
                output_dir, f"{side}_{self.transform_name}{self.file_end}"
            )
            self._write(sample, f"{side}_{self.transform_name}", output_path)

    @abc.abstractmethod
    def _write(self, sample: Dict[str, Any], key: str, output_path: str):
        pass


class PolarSerializer(Serializer):
    """
    Read and write outputs of polar_transform.
    For each side this corresponds to a list of dictionaries with the following keys:
        - "label": str is the label "external" or "internal",
        - "slice_idx": int is the index of the axial slice,
        - "polar_pt": Tensor is the tensor wrapping the 3D polar image,
        - "center_pt": array is the array of the 3 spatial coordinates of the center.
    Each side is also associated to a meta dict including the parameters of the polar transform.
    """

    def __init__(self, dir_path: str):
        super().__init__(
            dir_path,
            "polar",
            file_ext=None,
            monai_reader=Compose(
                [
                    LoadMetaJSONd(keys=["left_polar", "right_polar"], parent_dir=False),
                    LoadPolarDird(keys=["left_polar", "right_polar"]),
                ],
            ),
        )

    def _write(self, sample: Dict[str, Any], key: str, output_path: str):
        makedirs(output_path, exist_ok=True)
        polar_list = sample[key]
        for polar_dict in polar_list:
            label_name = polar_dict["label"]
            slice_idx = polar_dict["slice_idx"]
            polar_np = polar_dict["polar_pt"].squeeze(0).numpy()
            center_np = polar_dict["center_pt"].numpy()
            np.save(
                path.join(
                    output_path, f"label-{label_name}_slice-{slice_idx}_polar.npy"
                ),
                polar_np,
            )
            np.save(
                path.join(
                    output_path, f"label-{label_name}_slice-{slice_idx}_center.npy"
                ),
                center_np,
            )
        write_json(
            sample[f"{key}_meta_dict"],
            path.join(output_path, "parameters.json"),
        )


class HeatmapSerializer(Serializer):
    """
    Read and write outputs of heatmap_transform.
    For each side this corresponds to a numpy array representing the heatmap.
    """

    def __init__(self, dir_path: str):
        super().__init__(
            dir_path=dir_path,
            transform_name="heatmap",
            file_ext="npy",
            monai_reader=LoadImaged(
                keys=["left_heatmap", "right_heatmap"], reader="numpyreader"
            ),
        )

    def _write(self, sample: Dict[str, Any], key: str, output_path: str):
        heatmap_np = sample[key].numpy()
        np.save(output_path, heatmap_np)


class CenterlineSerializer(Serializer):
    """
    Read and write outputs of centerline_transform.
    For each side this corresponds to a TSV file listing the centers of each label (internal or external).
    """

    def __init__(self, dir_path: str):
        super().__init__(
            dir_path=dir_path,
            transform_name="centerline",
            file_ext="tsv",
            monai_reader=LoadCSVd(
                keys=["left_centerline", "right_centerline"], sep="\t"
            ),
        )

    def _write(self, sample: Dict[str, Any], key: str, output_path: str):
        sample[key].to_csv(output_path, sep="\t", index=False)


class ContourSerializer(Serializer):
    """
    Read and write outputs of contour_transform.
    For each side this corresponds to a point cloud of size (2, M, 3).
    Description of each axis:
        - index 0 is the lumen, index 1 is the wall,
        - M is the number of points in the cloud,
        - spatial coords, ordered as Z, Y, X.
    """

    def __init__(self, dir_path: str):
        super().__init__(
            dir_path=dir_path,
            transform_name="contour",
            file_ext="tsv",
            monai_reader=Compose(
                [
                    LoadMetaJSONd(
                        keys=["left_contour", "right_contour"], parent_dir=True
                    ),
                    LoadCSVd(keys=["left_contour", "right_contour"], sep="\t"),
                ],
            ),
        )

    def _write(self, sample: Dict[str, Any], key: str, output_path: str):
        sample[key].to_csv(output_path, sep="\t", index=False)
        write_json(
            sample[f"{key}_meta_dict"],
            path.join(path.dirname(output_path), "parameters.json"),
        )


class SegmentationSerializer(Serializer):
    """
    Read and write outputs of contour_transform.
    For each side this corresponds to a point cloud of size (2, M, 3).
    Description of each axis:
        - index 0 is the lumen, index 1 is the wall,
        - M is the number of points in the cloud,
        - spatial coords, ordered as Z, Y, X.
    """

    def __init__(self, dir_path: str):
        super().__init__(
            dir_path=dir_path,
            transform_name="segmentation",
            file_ext="npy",
            monai_reader=Compose(
                [
                    LoadImaged(
                        keys=[
                            "left_lumen_segmentation",
                            "left_wall_segmentation",
                            "right_lumen_segmentation",
                            "right_wall_segmentation",
                        ],
                        reader="numpyreader",
                    )
                ],
            ),
        )
        self.object_list = ["lumen", "wall"]

    def add_path(self, sample_list: List[Dict[str, str]]):
        for sample in sample_list:
            output_dir = path.join(
                self.dir_path,
                sample["participant_id"],
                f"{self.transform_name}_transform",
            )
            for side in self.side_list:
                for object_name in self.object_list:
                    sample[f"{side}_{object_name}_{self.transform_name}"] = path.join(
                        output_dir,
                        f"{side}_{object_name}_{self.transform_name}{self.file_end}",
                    )

    def write(self, sample: Dict[str, Any]):
        output_dir = path.join(
            self.dir_path, sample["participant_id"], f"{self.transform_name}_transform"
        )
        makedirs(output_dir, exist_ok=True)

        for side in self.side_list:
            for object_name in self.object_list:
                output_path = path.join(
                    output_dir,
                    f"{side}_{object_name}_{self.transform_name}{self.file_end}",
                )
                self._write(
                    sample, f"{side}_{object_name}_{self.transform_name}", output_path
                )

    def _write(self, sample: Dict[str, Any], key: str, output_path: str):
        heatmap_np = sample[key]
        np.save(output_path, heatmap_np)


class RawReader:
    """
    Reads raw images in DICOM or MHD format.
    """

    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.parameters = self.compute_raw_description(dir_path)
        self.monai_reader = Compose(self.get_transforms())

    def get_transforms(self) -> List[MapTransform]:
        """
        Returns list of transforms used to load, rescale and reshape MRI volume.
        """
        transforms = [
            LoadImaged(keys=["image"], ensure_channel_first=True),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=self.parameters["lower_percentile_rescaler"],
                upper=self.parameters["upper_percentile_rescaler"],
                b_min=0,
                b_max=1,
                clip=True,
            ),
            Orientationd(keys=["image"], axcodes="SPL"),
        ]

        return transforms

    def find_participant_set(self) -> Set[str]:

        data_type = self.parameters["data_type"]

        if data_type == "dcm":
            participant_set = {
                participant_id
                for participant_id in listdir(self.dir_path)
                if path.isdir(path.join(self.dir_path, participant_id))
            }
        else:
            participant_set = {
                participant_id.split(".")[0]
                for participant_id in listdir(self.dir_path)
                if participant_id.endswith(f".{data_type}")
            }

        return participant_set

    def add_path(self, sample_list: List[Dict[str, str]]):

        data_type = self.parameters["data_type"]

        for sample in sample_list:
            participant_id = sample["participant_id"]
            if data_type == "dcm":
                img_path = path.join(self.dir_path, participant_id)
            else:
                img_path = path.join(self.dir_path, f"{participant_id}.{data_type}")
            sample["image"] = img_path

    @staticmethod
    def compute_raw_description(raw_dir: str) -> Dict[str, Any]:
        """
        Outputs the set of parameters necessary to process a directory containing raw data based
        on a JSON file containing necessary parameters.

        Args:
            raw_dir: path to the directory containing raw data.

        Returns:
            dictionary of parameters to process the corresponding directory.
        """
        raw_parameters = read_json(path.join(raw_dir, "parameters.json"))
        mandatory_args = {
            "lower_percentile_rescaler",
            "upper_percentile_rescaler",
        }
        intersection_args = set(raw_parameters.keys()) & mandatory_args
        if intersection_args != mandatory_args:
            missing_args = mandatory_args - intersection_args
            raise MissingRawArgException(
                f"Arguments to describe raw data are missing.\n"
                f"Please define {missing_args} in the parameters.json "
                f"of the raw directory."
            )

        # Find data type
        file_list = listdir(raw_dir)
        mhd_list = [filename for filename in file_list if filename.endswith(".mhd")]
        mha_list = [filename for filename in file_list if filename.endswith(".mha")]
        if len(mhd_list) > 0:
            raw_parameters["data_type"] = "mhd"
        elif len(mha_list) > 0:
            raw_parameters["data_type"] = "mha"
        else:
            raw_parameters["data_type"] = "dcm"

        raw_parameters["dir"] = raw_dir

        return raw_parameters