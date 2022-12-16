from monai.transforms import (
    ScaleIntensityRangePercentilesd,
    LoadImaged,
    Transform,
    AddChanneld,
)
from os import path, listdir
from monai.data.image_reader import ITKReader
from monai.data import Dataset, CacheDataset
from carotid.utils.transforms import BuildEmptyLabelsd
from typing import List


def get_loader_transforms(
    lower_percentile_rescaler: float = 1,
    upper_percentile_rescaler: float = 99,
) -> List[Transform]:
    """
    Returns list of transforms used to load, rescale and reshape MRI volume.

    Args:
        lower_percentile_rescaler: lower percentile used to rescale intensities
        upper_percentile_rescaler: higher percentile used to rescale intensities
    """
    loader = LoadImaged(keys=["img"])
    loader.register(ITKReader(reverse_indexing=True))
    return [
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


def build_dataset(
    raw_dir: str,
    participant_list: List[str] = None,
    lower_percentile_rescaler: float = 1,
    upper_percentile_rescaler: float = 99,
) -> Dataset:

    if participant_list is None:
        participant_list = [
            participant_id
            for participant_id in listdir(raw_dir)
            if path.isdir(path.join(raw_dir, participant_id))
        ]

    sample_list = [
        {"img": path.join(raw_dir, participant_id), "participant_id": participant_id}
        for participant_id in participant_list
    ]
    return CacheDataset(
        sample_list,
        transform=get_loader_transforms(
            lower_percentile_rescaler, upper_percentile_rescaler
        ),
    )
