from carotid.heatmap_transform.utils import UNetPredictor
from carotid.centerline_transform.utils import OnePassExtractor
from carotid.polar_transform.utils import PolarTransform
from carotid.segmentation_transform.utils import SegmentationTransform
from os import path, makedirs
from carotid.utils import (
    write_json,
    read_and_fill_default_toml,
    build_dataset,
    check_device,
    SegmentationSerializer,
)
import toml
from typing import List

pipeline_dir = path.dirname(path.realpath(__file__))
list_transforms = [
    "heatmap_transform",
    "centerline_transform",
    "polar_transform",
    "segmentation_transform",
]


def apply_transform(
    raw_dir: str,
    heatmap_model_dir: str,
    segmentation_model_dir: str,
    output_dir: str,
    config_path: str = None,
    participant_list: List[str] = None,
    device: str = None,
):
    # Read parameters
    device = check_device(device=device)

    if config_path is None:
        config_dir = dict()
    else:
        config_dir = toml.load(config_path)

    for transform in list_transforms:
        if transform not in config_dir:
            config_dir[transform] = dict()
        config_dir[transform] = read_and_fill_default_toml(
            config_dir[transform],
            path.join(pipeline_dir, "..", transform, "default_args.toml"),
        )
        config_dir[transform]["raw_dir"] = raw_dir
        config_dir[transform]["device"] = device.type
        config_dir[transform]["dir"] = output_dir

    config_dir["heatmap_transform"]["model_dir"] = heatmap_model_dir
    config_dir["segmentation_transform"]["model_dir"] = segmentation_model_dir

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    write_json(config_dir, path.join(output_dir, "pipeline_parameters.json"))

    # Transforms
    unet_predictor = UNetPredictor(parameters=config_dir["heatmap_transform"])
    centerline_extractor = OnePassExtractor(
        parameters=config_dir["centerline_transform"]
    )
    polar_transform = PolarTransform(parameters=config_dir["polar_transform"])
    segmentation_transform = SegmentationTransform(
        parameters=config_dir["segmentation_transform"]
    )

    dataset = build_dataset(
        raw_dir=raw_dir,
        participant_list=participant_list,
    )
    serializer = SegmentationSerializer(
        dir_path=output_dir,
    )

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Pipeline transform {participant_id}...")
        sample = unet_predictor(sample)
        sample = centerline_extractor(sample)
        sample = polar_transform(sample)
        sample = segmentation_transform(sample)
        serializer.write(sample)
