from monai.transforms import Compose
from monai.data import Dataset, CacheDataset
from typing import List, Dict, Any, Set

from .raw_utils import RawLogger
from .heatmap_utils import HeatmapLogger
from .centerline_utils import CenterlineLogger
from .template import Logger


def compute_logger_list(
    raw_parameters: Dict[str, Any] = None,
    heatmap_parameters: Dict[str, Any] = None,
    centerline_parameters: Dict[str, Any] = None,
):
    logger_list = list()
    if raw_parameters is not None:
        logger_list.append(RawLogger(raw_parameters))

    if heatmap_parameters is not None:
        logger_list.append(HeatmapLogger(heatmap_parameters))

    if centerline_parameters is not None:
        logger_list.append(CenterlineLogger(centerline_parameters))


def compute_sample_list(
    logger_list: List[Logger], participant_set: Set[str] = None
) -> List[Dict[str, str]]:

    for logger in logger_list:
        if participant_set is not None:
            participant_set = participant_set & logger.find_participant_set()
        else:
            participant_set = logger.find_participant_set()

    sample_list = [
        {"participant_id": participant_id} for participant_id in participant_set
    ]
    print(sample_list)

    # Add the useful information for all existing inputs
    for logger in logger_list:
        logger.add_path(sample_list)

    print(sample_list)
    return sample_list


def build_dataset(
    logger_list: List[Logger],
    participant_list: List[str] = None,
) -> Dataset:

    sample_list = compute_sample_list(
        logger_list=logger_list,
        participant_set=set(participant_list) if participant_list is not None else None,
    )

    transform_list = list()
    for logger in logger_list:
        transform_list += logger.get_transforms()

    return CacheDataset(sample_list, transform=Compose(transform_list))
