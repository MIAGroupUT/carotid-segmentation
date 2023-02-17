from .utils import ContourTransform
from os import path, makedirs
from carotid.utils import (
    read_json,
    write_json,
    read_and_fill_default_toml,
    build_dataset,
    check_device,
    ContourSerializer,
)
from typing import List

pipeline_dir = path.dirname(path.realpath(__file__))


def apply_transform(
    output_dir: str,
    model_dir: str,
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

    # Read global default args
    contour_parameters = read_and_fill_default_toml(
        config_path, path.join(pipeline_dir, "default_args.toml")
    )

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    contour_parameters["raw_dir"] = raw_dir
    contour_parameters["polar_dir"] = polar_dir
    contour_parameters["model_dir"] = model_dir
    contour_parameters["device"] = device.type
    write_json(contour_parameters, path.join(output_dir, "contour_parameters.json"))

    contour_transform = ContourTransform(
        contour_parameters,
    )

    dataset = build_dataset(
        raw_dir=raw_dir,
        polar_dir=polar_dir,
        participant_list=participant_list,
    )
    serializer = ContourSerializer(output_dir)

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Contour transform {participant_id}...")

        predicted_sample = contour_transform(sample)
        serializer.write(predicted_sample)
