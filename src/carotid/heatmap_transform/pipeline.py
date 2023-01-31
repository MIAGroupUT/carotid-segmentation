from .utils import UNetPredictor
from os import path, makedirs
from carotid.utils import (
    read_json,
    write_json,
    read_and_fill_default_toml,
    build_dataset,
    check_device,
    compute_raw_description,
    RawLogger,
    HeatmapLogger,
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
    raw_parameters = compute_raw_description(raw_dir)
    raw_logger = RawLogger(raw_parameters)

    model_parameters = read_json(path.join(model_dir, "parameters.json"))  # TODO remove

    # Read global default args
    heatmap_parameters = read_and_fill_default_toml(
        config_path, path.join(pipeline_dir, "default_args.toml")
    )

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    heatmap_parameters["raw_dir"] = raw_dir
    heatmap_parameters["model_dir"] = model_dir
    heatmap_parameters["dir"] = output_dir
    heatmap_logger = HeatmapLogger(heatmap_parameters)
    write_json(heatmap_parameters, path.join(output_dir, "heatmap_parameters.json"))

    unet_predictor = UNetPredictor(
        model_dir=model_dir,
        roi_size=heatmap_parameters["roi_size"],
        flip_z=model_parameters["z_orientation"] != "down",  # Remove
        spacing=raw_parameters["spacing_required"],
        device=device,
    )
    dataset = build_dataset(
        [raw_logger],
        participant_list=participant_list,
    )

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Heatmap transform {participant_id}...")

        predicted_sample = unet_predictor(sample)
        heatmap_logger.write(predicted_sample)
