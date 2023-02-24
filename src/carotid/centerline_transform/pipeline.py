from .utils import OnePassExtractor
from os import path, makedirs
from carotid.utils import (
    read_json,
    write_json,
    read_and_fill_default_toml,
    check_transform_presence,
    build_dataset,
    CenterlineSerializer,
)
from typing import List

transform_name = path.basename(path.dirname(path.realpath(__file__)))


def apply_transform(
    output_dir: str,
    heatmap_dir: str = None,
    config_path: str = None,
    participant_list: List[str] = None,
    force: bool = False,
):
    # Read parameters
    if heatmap_dir is None:
        heatmap_dir = output_dir

    pipeline_parameters = read_json(path.join(heatmap_dir, "parameters.json"))

    # Read global default args
    centerline_parameters = read_and_fill_default_toml(config_path)[transform_name]

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    check_transform_presence(output_dir, transform_name, force=force)

    centerline_parameters["dir"] = output_dir
    centerline_parameters["heatmap_dir"] = heatmap_dir
    pipeline_parameters[transform_name] = centerline_parameters
    write_json(pipeline_parameters, path.join(output_dir, "parameters.json"))

    centerline_extractor = OnePassExtractor(centerline_parameters)

    dataset = build_dataset(
        heatmap_dir=heatmap_dir,
        participant_list=participant_list,
    )
    serializer = CenterlineSerializer(output_dir)

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Centerline transform {participant_id}...")

        centerline_dict = centerline_extractor(sample)
        serializer.write(centerline_dict)
