from .utils import OnePassExtractor
from os import path, makedirs
from carotid.utils import (
    check_device,
    write_json,
    read_and_fill_default_toml,
    build_dataset,
    HeatmapLogger,
    CenterlineLogger,
)
from typing import List

pipeline_dir = path.dirname(path.realpath(__file__))


def apply_transform(
    output_dir: str,
    input_dir: str = None,
    config_path: str = None,
    participant_list: List[str] = None,
    device: str = None,
):
    # Read parameters
    device = check_device(device=device)
    if input_dir is None:
        input_dir = output_dir

    # Read global default args
    pipeline_parameters = read_and_fill_default_toml(
        config_path, path.join(pipeline_dir, "default_args.toml")
    )
    heatmap_logger = HeatmapLogger({"dir": input_dir})

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    pipeline_parameters["dir"] = output_dir
    write_json(pipeline_parameters, path.join(output_dir, "centerline_parameters.json"))
    centerline_logger = CenterlineLogger(pipeline_parameters)

    centerline_extractor = OnePassExtractor(
        step_size=pipeline_parameters["step_size"],
        threshold=pipeline_parameters["threshold"],
    )

    dataset = build_dataset(
        [heatmap_logger],
        participant_list=participant_list,
    )

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Centerline transform {participant_id}...")

        centerline_dict = centerline_extractor(sample)
        centerline_logger.write(centerline_dict)
