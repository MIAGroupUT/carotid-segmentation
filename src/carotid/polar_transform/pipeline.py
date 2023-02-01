from .utils import PolarTransform
from os import path, makedirs
from carotid.utils import (
    read_json,
    write_json,
    read_and_fill_default_toml,
    build_dataset,
    check_device,
    compute_raw_description,
    RawLogger,
    CenterlineLogger,
    PolarLogger,
)
from typing import List

pipeline_dir = path.dirname(path.realpath(__file__))


def apply_transform(
    raw_dir: str,
    output_dir: str,
    config_path: str = None,
    participant_list: List[str] = None,
    device: str = None,
):
    # Read parameters
    device = check_device(device=device)

    raw_parameters = compute_raw_description(raw_dir)
    raw_logger = RawLogger(raw_parameters)
    centerline_logger = CenterlineLogger({"dir": output_dir})

    # Read global default args
    polar_parameters = read_and_fill_default_toml(
        config_path, path.join(pipeline_dir, "default_args.toml")
    )

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    polar_parameters["raw_dir"] = raw_dir
    polar_parameters["dir"] = output_dir
    polar_logger = PolarLogger(polar_parameters)
    write_json(polar_parameters, path.join(output_dir, "polar_parameters.json"))

    polar_transform = PolarTransform(
        n_angles=polar_parameters["n_angles"],
        cartesian_ray=polar_parameters["cartesian_ray"],
        polar_ray=polar_parameters["polar_ray"],
        length=polar_parameters["length"],
    )

    dataset = build_dataset(
        [raw_logger, centerline_logger],
        participant_list=participant_list,
    )

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Heatmap transform {participant_id}...")

        predicted_sample = polar_transform(sample)
        polar_logger.write(predicted_sample)
