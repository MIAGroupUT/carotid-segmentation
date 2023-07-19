from .utils import ContourTransform
from os import path, makedirs
from carotid.utils import (
    read_json,
    write_json,
    read_and_fill_default_toml,
    check_transform_presence,
    build_dataset,
    check_device,
    ContourSerializer,
)
from typing import List
from logging import getLogger

transform_name = f"{path.basename(path.dirname(path.realpath(__file__)))}_transform"


logger = getLogger("carotid")


def apply_transform(
    output_dir: str,
    model_dir: str,
    polar_dir: str = None,
    config_path: str = None,
    participant_list: List[str] = None,
    device: str = None,
    force: bool = False,
):
    # Read parameters
    device = check_device(device=device)
    if polar_dir is None:
        polar_dir = output_dir

    pipeline_parameters = read_json(path.join(polar_dir, "parameters.json"))

    # Read global default args
    contour_parameters = read_and_fill_default_toml(config_path)[transform_name]

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    check_transform_presence(output_dir, transform_name, force=force)

    contour_parameters["polar_dir"] = polar_dir
    contour_parameters["model_dir"] = model_dir
    contour_parameters["device"] = device.type
    pipeline_parameters[transform_name] = contour_parameters
    write_json(pipeline_parameters, path.join(output_dir, "parameters.json"))

    contour_transform = ContourTransform(
        contour_parameters,
    )

    dataset = build_dataset(
        polar_dir=polar_dir,
        participant_list=participant_list,
    )
    serializer = ContourSerializer(output_dir)

    for sample in dataset:
        participant_id = sample["participant_id"]
        logger.info(f"Contour transform {participant_id}...")

        predicted_sample = contour_transform(sample)
        serializer.write(predicted_sample)
