from .utils import PolarTransform
from os import path, makedirs
from carotid.utils import (
    read_json,
    write_json,
    read_and_fill_default_toml,
    check_transform_presence,
    build_dataset,
    PolarSerializer,
)
from typing import List

transform_name = f"{path.basename(path.dirname(path.realpath(__file__)))}_transform"


def apply_transform(
    output_dir: str,
    centerline_dir: str = None,
    config_path: str = None,
    participant_list: List[str] = None,
    force: bool = False,
):
    # Read parameters
    if centerline_dir is None:
        centerline_dir = output_dir

    pipeline_parameters = read_json(path.join(centerline_dir, "parameters.json"))
    raw_dir = pipeline_parameters["heatmap_transform"]["raw_dir"]

    # Read global default args
    polar_parameters = read_and_fill_default_toml(config_path)[transform_name]

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    check_transform_presence(output_dir, transform_name, force=force)

    polar_parameters["raw_dir"] = raw_dir
    polar_parameters["centerline_dir"] = centerline_dir
    polar_parameters["dir"] = output_dir
    pipeline_parameters[transform_name] = polar_parameters
    write_json(pipeline_parameters, path.join(output_dir, "parameters.json"))

    polar_transform = PolarTransform(polar_parameters)
    dataset = build_dataset(
        raw_dir=raw_dir,
        centerline_dir=centerline_dir,
        participant_list=participant_list,
    )
    serializer = PolarSerializer(output_dir)

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Polar transform {participant_id}...")

        predicted_sample = polar_transform(sample)
        serializer.write(predicted_sample)
