from .utils import SegmentationTransform
from os import path, makedirs
from carotid.utils import (
    read_json,
    write_json,
    read_and_fill_default_toml,
    build_dataset,
    SegmentationSerializer,
)
from typing import List

pipeline_dir = path.dirname(path.realpath(__file__))


def apply_transform(
    output_dir: str,
    contour_dir: str = None,
    config_path: str = None,
    participant_list: List[str] = None,
):
    """
    This segmentation procedure over-segment in the right & anterior directions and under-segment
    in the left & posterior directions.
    """
    # Read parameters
    if contour_dir is None:
        contour_dir = output_dir

    centerline_parameters = read_json(path.join(contour_dir, "contour_parameters.json"))
    raw_dir = centerline_parameters["raw_dir"]

    # Read global default args
    segmentation_parameters = read_and_fill_default_toml(
        config_path, path.join(pipeline_dir, "default_args.toml")
    )

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    segmentation_parameters["raw_dir"] = raw_dir
    segmentation_parameters["contour_dir"] = contour_dir
    segmentation_parameters["dir"] = output_dir
    write_json(
        segmentation_parameters, path.join(output_dir, "segmentation_parameters.json")
    )

    polar_transform = SegmentationTransform(segmentation_parameters)
    dataset = build_dataset(
        raw_dir=raw_dir,
        contour_dir=contour_dir,
        participant_list=participant_list,
    )
    serializer = SegmentationSerializer(output_dir)

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Segmentation transform {participant_id}...")

        predicted_sample = polar_transform(sample)
        serializer.write(predicted_sample)
