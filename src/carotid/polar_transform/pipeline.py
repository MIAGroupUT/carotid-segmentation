from .utils import PolarTransform
from os import path, makedirs
from carotid.utils import (
    read_json,
    write_json,
    read_and_fill_default_toml,
    build_dataset,
    check_device,
    compute_raw_description,
    PolarSerializer,
)
from typing import List

pipeline_dir = path.dirname(path.realpath(__file__))


def apply_transform(
    output_dir: str,
    centerline_dir: str = None,
    config_path: str = None,
    participant_list: List[str] = None,
    device: str = None,
):
    # Read parameters
    device = check_device(device=device)
    if centerline_dir is None:
        centerline_dir = output_dir

    centerline_parameters = read_json(
        path.join(centerline_dir, "centerline_parameters.json")
    )
    raw_dir = centerline_parameters["raw_dir"]

    raw_parameters = compute_raw_description(raw_dir)

    # Read global default args
    polar_parameters = read_and_fill_default_toml(
        config_path, path.join(pipeline_dir, "default_args.toml")
    )

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    polar_parameters["raw_dir"] = raw_dir
    polar_parameters["centerline_dir"] = centerline_dir
    polar_parameters["dir"] = output_dir
    polar_logger = PolarSerializer(polar_parameters)
    write_json(polar_parameters, path.join(output_dir, "polar_parameters.json"))

    polar_transform = PolarTransform(
        n_angles=polar_parameters["n_angles"],
        cartesian_ray=polar_parameters["cartesian_ray"],
        polar_ray=polar_parameters["polar_ray"],
        length=polar_parameters["length"],
    )

    dataset = build_dataset(
        raw_parameters=raw_parameters,
        centerline_parameters={"dir": centerline_dir},
        participant_list=participant_list,
    )

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Polar transform {participant_id}...")

        predicted_sample = polar_transform(sample)
        polar_logger.write(predicted_sample)
