from .utils import SegmentationTransform
from os import path, makedirs
from carotid.utils import (
    read_json,
    write_json,
    read_and_fill_default_toml,
    build_dataset,
    check_device,
    compute_raw_description,
    SegmentationSerializer,
)
from typing import List

pipeline_dir = path.dirname(path.realpath(__file__))


def apply_transform(
    output_dir: str,
    polar_dir: str = None,
    config_path: str = None,
    participant_list: List[str] = None,
    device: str = None,
):
    # Read parameters
    device = check_device(device=device)
    if polar_dir is None:
        polar_dir = output_dir

    polar_parameters = read_json(path.join(polar_dir, "polar_parameters.json"))
    raw_dir = polar_parameters["raw_dir"]
    centerline_dir = polar_parameters["centerline_dir"]

    raw_parameters = compute_raw_description(raw_dir)

    # Read global default args
    segmentation_parameters = read_and_fill_default_toml(
        config_path, path.join(pipeline_dir, "default_args.toml")
    )

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    segmentation_parameters["raw_dir"] = raw_dir
    segmentation_parameters["polar_dir"] = polar_dir
    segmentation_parameters["centerline_dir"] = centerline_dir
    segmentation_parameters["dir"] = output_dir
    segmentation_logger = SegmentationSerializer(segmentation_parameters)
    write_json(segmentation_parameters, path.join(output_dir, "polar_parameters.json"))

    segmentation_transform = SegmentationTransform(segmentation_parameters)

    dataset = build_dataset(
        raw_parameters=raw_parameters,
        polar_parameters={"dir": polar_dir},
        participant_list=participant_list,
    )

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Heatmap transform {participant_id}...")

        predicted_sample = segmentation_transform(sample)
        segmentation_logger.write(predicted_sample)
