from monai.transforms import Compose
from monai.data import Dataset, CacheDataset
from typing import List, Dict, Any, Set

from .raw_utils import RawSerializer
from .heatmap_utils import HeatmapSerializer
from .centerline_utils import CenterlineSerializer
from .polar_utils import PolarSerializer
from .template import Serializer


def compute_serializer_list(
    raw_parameters: Dict[str, Any] = None,
    heatmap_parameters: Dict[str, Any] = None,
    centerline_parameters: Dict[str, Any] = None,
    polar_parameters: Dict[str, Any] = None,
):
    serializer_list = list()
    if raw_parameters is not None:
        serializer_list.append(RawSerializer(raw_parameters))

    if heatmap_parameters is not None:
        serializer_list.append(HeatmapSerializer(heatmap_parameters))

    if centerline_parameters is not None:
        serializer_list.append(CenterlineSerializer(centerline_parameters))

    if polar_parameters is not None:
        serializer_list.append(PolarSerializer(polar_parameters))

    return serializer_list


def compute_sample_list(
    serializer_list: List[Serializer], participant_set: Set[str] = None
) -> List[Dict[str, str]]:

    for serializer in serializer_list:
        if participant_set is not None:
            participant_set = participant_set & serializer.find_participant_set()
        else:
            participant_set = serializer.find_participant_set()

    sample_list = [
        {"participant_id": participant_id} for participant_id in participant_set
    ]
    print(sample_list)

    # Add the useful information for all existing inputs
    for serializer in serializer_list:
        serializer.add_path(sample_list)

    print(sample_list)
    return sample_list


def build_dataset(
    raw_parameters: Dict[str, Any] = None,
    heatmap_parameters: Dict[str, Any] = None,
    centerline_parameters: Dict[str, Any] = None,
    polar_parameters: Dict[str, Any] = None,
    participant_list: List[str] = None,
) -> Dataset:

    serializer_list = compute_serializer_list(
        raw_parameters=raw_parameters,
        heatmap_parameters=heatmap_parameters,
        centerline_parameters=centerline_parameters,
        polar_parameters=polar_parameters,
    )

    sample_list = compute_sample_list(
        serializer_list=serializer_list,
        participant_set=set(participant_list) if participant_list is not None else None,
    )

    transform_list = list()
    for serializer in serializer_list:
        transform_list += serializer.get_transforms()

    return CacheDataset(sample_list, transform=Compose(transform_list))
