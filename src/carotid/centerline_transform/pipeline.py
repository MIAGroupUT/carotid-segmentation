from utils import UNetPredictor, get_centerline_extractor, save_mevislab_markerfile, side_list, save_heatmaps
from os import path, makedirs
import toml
from carotid.utils import read_json, build_dataset
from typing import List


def apply_transform(
    raw_dir: str,
    model_dir: str,
    config_path: str,
    output_dir: str,
    participant_list: List[str] = None,
    device: str = "cuda",
):
    raw_parameters = read_json(path.join(raw_dir, "parameters.json"))
    model_parameters = read_json(path.join(model_dir, "parameters.json"))
    pipeline_parameters = toml.load(config_path)
    makedirs(output_dir, exist_ok=True)

    unet_predictor = UNetPredictor(
        model_dir=model_dir,
        roi_size=pipeline_parameters["roi_size"],
        flip_z=raw_parameters["z_orientation"] != model_parameters["z_orientation"],
        spacing=raw_parameters["spacing_required"],
        device=device
    )
    centerline_extractor = get_centerline_extractor(
        **pipeline_parameters
    )

    dataset = build_dataset(
        raw_dir,
        participant_list=participant_list,
        lower_percentile_rescaler=pipeline_parameters["lower_percentile_rescaler"],
        upper_percentile_rescaler=pipeline_parameters["upper_percentile_rescaler"],
    )

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Transform {participant_id}...")
        participant_path = path.join(output_dir, participant_id, "centerline_transform")
        makedirs(participant_path, exist_ok=True)

        predicted_sample = unet_predictor(sample)

        for side in side_list:
            for label in ["internal", "external"]:
                save_heatmaps(predicted_sample, output_dir, side, label)

        centerline_dict = centerline_extractor(predicted_sample)

        for side in side_list:
            for label in ["internal", "external"]:
                save_mevislab_markerfile(
                    centerline_dict[side][label],
                    path.join(participant_path, f"{side}_{label}.xml")
                )

