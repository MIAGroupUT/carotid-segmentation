from .utils import UNetPredictor
from os import path, makedirs
from carotid.utils import (
    write_json,
    read_and_fill_default_toml,
    build_dataset,
    check_device,
    HeatmapSerializer,
)
from typing import List

pipeline_dir = path.dirname(path.realpath(__file__))


def apply_transform(
    raw_dir: str,
    model_dir: str,
    output_dir: str,
    config_path: str = None,
    participant_list: List[str] = None,
    device: str = None,
):
    # Read parameters
    device = check_device(device=device)

    # Read global default args
    heatmap_parameters = read_and_fill_default_toml(
        config_path, path.join(pipeline_dir, "default_args.toml")
    )

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    heatmap_parameters["raw_dir"] = raw_dir
    heatmap_parameters["model_dir"] = model_dir
    heatmap_parameters["device"] = device.type
    write_json(heatmap_parameters, path.join(output_dir, "heatmap_parameters.json"))

    unet_predictor = UNetPredictor(parameters=heatmap_parameters)
    dataset = build_dataset(
        raw_dir=raw_dir,
        participant_list=participant_list,
    )
    serializer = HeatmapSerializer(output_dir)

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Heatmap transform {participant_id}...")

        predicted_sample = unet_predictor(sample)
        serializer.write(predicted_sample)
