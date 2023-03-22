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
    unet_predictor = UNetPredictor(parameters=pipeline_parameters["heatmap_transform"])
    centerline_extractor = OnePassExtractor(
        parameters=pipeline_parameters["centerline_transform"]
    )
    polar_transform = PolarTransform(parameters=pipeline_parameters["polar_transform"])
    contour_transform = ContourTransform(parameters=pipeline_parameters["contour_transform"])
    segmentation_transform = SegmentationTransform(
        parameters=pipeline_parameters["segmentation_transform"]
    )

    dataset = build_dataset(
        raw_dir=raw_dir,
        participant_list=participant_list,
    )
    serializers_list = [SegmentationSerializer(dir_path=output_dir)]
    if write_heatmap:
        serializers_list.append(HeatmapSerializer(dir_path=output_dir))
    if write_centerline:
        serializers_list.append(CenterlineSerializer(dir_path=output_dir))
    if write_polar:
        serializers_list.append(PolarSerializer(dir_path=output_dir))
    if write_contours:
        serializers_list.append(ContourSerializer(dir_path=output_dir))

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Pipeline transform {participant_id}...")
        sample = unet_predictor(sample)
        sample = centerline_extractor(sample)
        sample = polar_transform(sample)
        sample = contour_transform(sample)
        sample = segmentation_transform(sample)
        for serializer in serializers_list:
            serializer.write(sample)
