import shutil

from carotid.transform.heatmap.utils import UNetPredictor
from carotid.transform.centerline.utils import OnePassExtractor
from carotid.transform.polar.utils import PolarTransform
from carotid.transform.contour.utils import ContourTransform
from carotid.transform.segmentation.utils import SegmentationTransform
from os import path, makedirs
from carotid.utils import (
    write_json,
    read_and_fill_default_toml,
    build_dataset,
    check_device,
    check_transform_presence,
    HeatmapSerializer,
    CenterlineSerializer,
    PolarSerializer,
    ContourSerializer,
    SegmentationSerializer,
)
from typing import List
from logging import getLogger

logger = getLogger("carotid")

pipeline_dir = path.dirname(path.realpath(__file__))
list_transforms = [
    "heatmap_transform",
    "centerline_transform",
    "polar_transform",
    "contour_transform",
    "segmentation_transform",
]


def apply_transform(
    raw_dir: str,
    heatmap_model_dir: str,
    contour_model_dir: str,
    output_dir: str,
    config_path: str = None,
    participant_list: List[str] = None,
    device: str = None,
    force: bool = False,
    write_heatmap: bool = False,
    write_centerline: bool = False,
    write_polar: bool = False,
    write_contours: bool = False,
):
    # Read parameters
    device = check_device(device=device)

    pipeline_parameters = read_and_fill_default_toml(config_path)

    pipeline_parameters["heatmap_transform"]["model_dir"] = heatmap_model_dir
    pipeline_parameters["heatmap_transform"]["device"] = device.type
    pipeline_parameters["contour_transform"]["model_dir"] = contour_model_dir
    pipeline_parameters["contour_transform"]["device"] = device.type

    # Write parameters
    if force and path.exists(output_dir):
        shutil.rmtree(output_dir)

    makedirs(output_dir, exist_ok=True)
    for transform_name in list_transforms:
        check_transform_presence(output_dir, transform_name, force=force)

    write_json(pipeline_parameters, path.join(output_dir, "parameters.json"))

    # Transforms
    transform_dict = dict()
    transform_dict["heatmap_transform"] = UNetPredictor(
        parameters=pipeline_parameters["heatmap_transform"]
    )
    transform_dict["centerline_transform"] = OnePassExtractor(
        parameters=pipeline_parameters["centerline_transform"]
    )
    transform_dict["polar_transform"] = PolarTransform(
        parameters=pipeline_parameters["polar_transform"]
    )
    transform_dict["contour_transform"] = ContourTransform(
        parameters=pipeline_parameters["contour_transform"]
    )
    transform_dict["segmentation_transform"] = SegmentationTransform(
        parameters=pipeline_parameters["segmentation_transform"]
    )

    dataset = build_dataset(
        raw_dir=raw_dir,
        participant_list=participant_list,
    )

    serializers_dict = {
        "segmentation_transform": SegmentationSerializer(dir_path=output_dir)
    }
    if write_heatmap:
        serializers_dict["heatmap_transform"] = HeatmapSerializer(dir_path=output_dir)
    if write_centerline:
        serializers_dict["centerline_transform"] = CenterlineSerializer(
            dir_path=output_dir
        )
    if write_polar:
        serializers_dict["polar_transform"] = PolarSerializer(dir_path=output_dir)
    if write_contours:
        serializers_dict["contour_transform"] = ContourSerializer(dir_path=output_dir)

    for sample in dataset:
        participant_id = sample["participant_id"]
        logger.info(f"Pipeline transform {participant_id}...")
        for transform_name in list_transforms:
            sample = transform_dict[transform_name](sample)
            if transform_name in serializers_dict.keys():
                serializers_dict[transform_name].write(sample)
