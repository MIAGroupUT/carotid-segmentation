from .utils import (
    UNetPredictor,
    get_centerline_extractor,
    save_mevislab_markerfile,
    side_list,
    save_heatmaps,
)
from os import path, makedirs
from carotid.utils import (
    read_json,
    write_json,
    read_and_fill_default_toml,
    build_dataset,
)
from typing import List

pipeline_dir = path.dirname(path.realpath(__file__))


def apply_transform(
    raw_dir: str,
    model_dir: str,
    config_path: str,
    output_dir: str,
    participant_list: List[str] = None,
    device: str = "cuda",
):
    # Read parameters
    raw_parameters = read_json(path.join(raw_dir, "parameters.json"))
    model_parameters = read_json(path.join(model_dir, "parameters.json"))

    # Read global default args
    pipeline_parameters = read_and_fill_default_toml(
        config_path, path.join(pipeline_dir, "default_args.toml")
    )
    # Read post-processing default args
    method = pipeline_parameters["post_processing"]["method"]
    pipeline_parameters = read_and_fill_default_toml(
        pipeline_parameters,
        path.join(pipeline_dir, "post_processing", f"{method}.toml"),
    )

    # Write parameters
    makedirs(output_dir, exist_ok=True)
    pipeline_parameters["raw_dir"] = raw_dir
    pipeline_parameters["centerline_transform"]["model_dir"] = model_dir
    write_json(pipeline_parameters, path.join(output_dir, "parameters.json"))

    centerline_parameters = pipeline_parameters["centerline_transform"]

    unet_predictor = UNetPredictor(
        model_dir=model_dir,
        roi_size=centerline_parameters["roi_size"],
        flip_z=model_parameters["z_orientation"] != "down",
        spacing=raw_parameters["spacing_required"],
        device=device,
    )
    centerline_extractor = get_centerline_extractor(
        **pipeline_parameters["post_processing"]
    )

    dataset = build_dataset(
        raw_dir,
        participant_list=participant_list,
        lower_percentile_rescaler=centerline_parameters["lower_percentile_rescaler"],
        upper_percentile_rescaler=centerline_parameters["upper_percentile_rescaler"],
        z_flip=raw_parameters["z_orientation"] != "down",
        data_type=raw_parameters["data_type"],
    )

    for sample in dataset:
        participant_id = sample["participant_id"]
        print(f"Centerline transform {participant_id}...")
        participant_path = path.join(output_dir, participant_id, "centerline_transform")
        makedirs(participant_path, exist_ok=True)

        predicted_sample = unet_predictor(sample)

        for side in side_list:
            for label in ["internal", "external"]:
                save_heatmaps(predicted_sample, participant_path, side, label)

        centerline_dict = centerline_extractor(predicted_sample)

        for side in side_list:
            for label in ["internal", "external"]:
                save_mevislab_markerfile(
                    centerline_dict[side][label],
                    path.join(participant_path, f"{side}_{label}_markers.xml"),
                )
