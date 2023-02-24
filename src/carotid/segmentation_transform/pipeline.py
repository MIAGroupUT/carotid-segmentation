from .utils import SegmentationTransform
from os import path, makedirs
from carotid.utils import (
    read_json,
    write_json,
    read_and_fill_default_toml,
    check_transform_presence,
    build_dataset,
    SegmentationSerializer,
)
from typing import List

transform_name = path.basename(path.dirname(path.realpath(__file__)))


def apply_transform(
    output_dir: str,
    contour_dir: str = None,
    config_path: str = None,
    participant_list: List[str] = None,
    force: bool = False,
):
    """
    This segmentation procedure over-segment in the right & anterior directions and under-segment
    in the left & posterior directions.
    """
    # Read parameters
    if contour_dir is None:
        contour_dir = output_dir

    pipeline_parameters = read_json(path.join(contour_dir, "parameters.json"))

    # Read global default args
    segmentation_parameters = read_and_fill_default_toml(config_path)

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    check_transform_presence(output_dir, transform_name, force=force)

    segmentation_parameters["contour_dir"] = contour_dir
    segmentation_parameters["dir"] = output_dir
    pipeline_parameters[transform_name] = segmentation_parameters
    write_json(pipeline_parameters, path.join(output_dir, "parameters.json"))

    segmentation_transform = SegmentationTransform(segmentation_parameters)
    dataset = build_dataset(
        contour_dir=contour_dir,
        participant_list=participant_list,
    )
    serializer = SegmentationSerializer(output_dir)

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Segmentation transform {participant_id}...")

        predicted_sample = segmentation_transform(sample)
        serializer.write(predicted_sample)
