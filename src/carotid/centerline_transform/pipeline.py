from .utils import OnePassExtractor
from os import path, makedirs
from carotid.utils import (
    read_json,
    check_device,
    write_json,
    read_and_fill_default_toml,
    build_dataset,
    CenterlineSerializer,
)
from typing import List

pipeline_dir = path.dirname(path.realpath(__file__))


def apply_transform(
    output_dir: str,
    heatmap_dir: str = None,
    config_path: str = None,
    participant_list: List[str] = None,
    device: str = None,
):
    # Read parameters
    device = check_device(device=device)
    if heatmap_dir is None:
        heatmap_dir = output_dir

    heatmap_parameters = read_json(path.join(heatmap_dir, "heatmap_parameters.json"))

    # Read global default args
    pipeline_parameters = read_and_fill_default_toml(
        config_path, path.join(pipeline_dir, "default_args.toml")
    )

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    pipeline_parameters["dir"] = output_dir
    pipeline_parameters["heatmap_dir"] = heatmap_dir
    pipeline_parameters["raw_dir"] = heatmap_parameters["raw_dir"]
    write_json(pipeline_parameters, path.join(output_dir, "centerline_parameters.json"))

    centerline_logger = CenterlineSerializer(pipeline_parameters)
    centerline_extractor = OnePassExtractor(pipeline_parameters)

    dataset = build_dataset(
        heatmap_parameters={"dir": heatmap_dir},
        participant_list=participant_list,
    )

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Centerline transform {participant_id}...")

        centerline_dict = centerline_extractor(sample)
        centerline_logger.write(centerline_dict)
