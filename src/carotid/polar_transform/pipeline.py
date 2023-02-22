from .utils import PolarTransform
from os import path, makedirs
from carotid.utils import (
    read_json,
    write_json,
    read_and_fill_default_toml,
    build_dataset,
    PolarSerializer,
)
from typing import List

pipeline_dir = path.dirname(path.realpath(__file__))


def apply_transform(
    output_dir: str,
    centerline_dir: str = None,
    config_path: str = None,
    participant_list: List[str] = None,
):
    # Read parameters
    if centerline_dir is None:
        centerline_dir = output_dir

    centerline_parameters = read_json(
        path.join(centerline_dir, "centerline_parameters.json")
    )
    raw_dir = centerline_parameters["raw_dir"]

    # Read global default args
    polar_parameters = read_and_fill_default_toml(
        config_path, path.join(pipeline_dir, "default_args.toml")
    )

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    polar_parameters["raw_dir"] = raw_dir
    polar_parameters["centerline_dir"] = centerline_dir
    polar_parameters["dir"] = output_dir
    write_json(polar_parameters, path.join(output_dir, "polar_parameters.json"))

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
