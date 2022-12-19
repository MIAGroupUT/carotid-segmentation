from monai.transforms import (
    ScaleIntensityRangePercentilesd,
    LoadImaged,
    Transform,
    AddChanneld,
    Flipd,
    Compose,
)
from os import path, listdir
from monai.data.image_reader import ITKReader
from monai.data import Dataset, CacheDataset
from carotid.utils.transforms import BuildEmptyLabelsd
from typing import List


def get_loader_transforms(
    lower_percentile_rescaler: float = 1,
    upper_percentile_rescaler: float = 99,
    z_flip: bool = False,
) -> List[Transform]:
    """
    Returns list of transforms used to load, rescale and reshape MRI volume.

    Args:
        lower_percentile_rescaler: lower percentile used to rescale intensities.
        upper_percentile_rescaler: higher percentile used to rescale intensities.
        z_flip: flip along the orthogonal direction to axial slices.
    """
    loader = LoadImaged(keys=["img"])
    loader.register(ITKReader(reverse_indexing=True))
    transforms = [
        loader,
        ScaleIntensityRangePercentilesd(
            keys=["img"],
            lower=lower_percentile_rescaler,
            upper=upper_percentile_rescaler,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        AddChanneld(keys=["img"]),
        BuildEmptyLabelsd(image_key="img"),
    ]
    if z_flip:
        transforms.append(Flipd(keys=["img"], spatial_axis=0))

    return transforms


def build_dataset(
    raw_dir: str,
    participant_list: List[str] = None,
    lower_percentile_rescaler: float = 1,
    upper_percentile_rescaler: float = 99,
    z_flip: bool = False,
    data_type: str = "dcm",
) -> Dataset:

    if participant_list is None or len(participant_list) == 0:
        if data_type == "dcm":
            participant_list = [
                participant_id
                for participant_id in listdir(raw_dir)
                if path.isdir(path.join(raw_dir, participant_id))
            ]
        else:
            participant_list = [
                participant_id.split(".")[0]
                for participant_id in listdir(raw_dir)
                if participant_id.endswith(".mhd")
            ]

    if data_type == "dcm":
        sample_list = [
            {
                "img": path.join(raw_dir, participant_id),
                "participant_id": participant_id,
            }
            for participant_id in participant_list
        ]
    else:
        sample_list = [
            {
                "img": path.join(raw_dir, f"{participant_id}.mhd"),
                "participant_id": participant_id,
            }
            for participant_id in participant_list
        ]

    return CacheDataset(
        sample_list,
        transform=Compose(
            get_loader_transforms(
                lower_percentile_rescaler, upper_percentile_rescaler, z_flip
            )
        ),
    )
