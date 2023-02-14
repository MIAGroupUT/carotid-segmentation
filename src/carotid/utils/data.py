from monai.transforms import Compose
from monai.data import Dataset, CacheDataset
from typing import List, Dict, Set

from .serializer import (
    HeatmapSerializer,
    CenterlineSerializer,
    PolarSerializer,
    ContourSerializer,
    Serializer,
    RawReader,
)


def compute_serializer_list(
    raw_dir: str = None,
    heatmap_dir: str = None,
    centerline_dir: str = None,
    polar_dir: str = None,
    contour_dir: str = None,
) -> List[Serializer]:
    serializer_list = list()
    if raw_dir is not None:
        serializer_list.append(RawReader(raw_dir))

    if heatmap_dir is not None:
        serializer_list.append(HeatmapSerializer(heatmap_dir))

    if centerline_dir is not None:
        serializer_list.append(CenterlineSerializer(centerline_dir))

    if polar_dir is not None:
        serializer_list.append(PolarSerializer(polar_dir))

    if contour_dir is not None:
        serializer_list.append(ContourSerializer(contour_dir))

    return serializer_list


def compute_sample_list(
    serializer_list: List[Serializer], participant_set: Set[str] = None
) -> List[Dict[str, str]]:

    for serializer in serializer_list:
        if participant_set is not None and len(participant_set) != 0:
            participant_set = participant_set & serializer.find_participant_set()
        else:
            participant_set = serializer.find_participant_set()

    sample_list = [
        {"participant_id": participant_id} for participant_id in participant_set
    ]

    # Add the useful information for all existing inputs
    for serializer in serializer_list:
        serializer.add_path(sample_list)

    return sample_list


# TODO: add pipeline_dir looking for all possible components of the pipeline in the same folder
def build_dataset(
    raw_dir: str = None,
    heatmap_dir: str = None,
    centerline_dir: str = None,
    polar_dir: str = None,
    contour_dir: str = None,
    participant_list: List[str] = None,
) -> Dataset:

    serializer_list = compute_serializer_list(
        raw_dir=raw_dir,
        heatmap_dir=heatmap_dir,
        centerline_dir=centerline_dir,
        polar_dir=polar_dir,
        contour_dir=contour_dir,
    )

    sample_list = compute_sample_list(
        serializer_list=serializer_list,
        participant_set=set(participant_list) if participant_list is not None else None,
    )

    transform_list = list()
    for serializer in serializer_list:
        transform_list.append(serializer.monai_reader)

    return CacheDataset(sample_list, transform=Compose(transform_list))
