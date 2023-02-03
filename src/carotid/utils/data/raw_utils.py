from monai.transforms import (
    ScaleIntensityRangePercentilesd,
    LoadImaged,
    Transform,
    AddChanneld,
)
from os import path, listdir
from monai.data.image_reader import ITKReader
from carotid.utils.logger import read_json
from typing import List, Dict, Any, Set
from .template import Serializer

from .errors import MissingRawArgException


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
    if len(mhd_list) > 0:
        raw_parameters["data_type"] = "mhd"
    else:
        raw_parameters["data_type"] = "dcm"

    raw_parameters["dir"] = raw_dir

    return raw_parameters


class RawSerializer(Serializer):
    def get_transforms(self) -> List[Transform]:
        """
        Returns list of transforms used to load, rescale and reshape MRI volume.

        Args:
            dictionary obtained by compute_raw_description.
        """
        loader = LoadImaged(keys=["img"])
        loader.register(
            ITKReader(reverse_indexing=True)
        )  # TODO: remove + add orientation
        transforms = [
            loader,
            ScaleIntensityRangePercentilesd(
                keys=["img"],
                lower=self.parameters["lower_percentile_rescaler"],
                upper=self.parameters["upper_percentile_rescaler"],
                b_min=0,
                b_max=1,
                clip=True,
            ),
            AddChanneld(keys=["img"]),
        ]

        return transforms

    def find_participant_set(self) -> Set[str]:

        raw_dir = self.parameters["dir"]
        data_type = self.parameters["data_type"]

        if data_type == "dcm":
            participant_set = {
                participant_id
                for participant_id in listdir(raw_dir)
                if path.isdir(path.join(raw_dir, participant_id))
            }
        else:
            participant_set = {
                participant_id.split(".")[0]
                for participant_id in listdir(raw_dir)
                if participant_id.endswith(".mhd")
            }

        return participant_set

    def add_path(self, sample_list: List[Dict[str, str]]):

        raw_dir = self.parameters["dir"]
        data_type = self.parameters["data_type"]

        for sample in sample_list:
            participant_id = sample["participant_id"]
            if data_type == "dcm":
                img_path = path.join(raw_dir, participant_id)
            else:
                img_path = path.join(raw_dir, f"{participant_id}.mhd")
            sample["img"] = img_path

    def write(self, sample: Dict[str, Any]):
        raise IOError("You are not supposed to write raw data.")
