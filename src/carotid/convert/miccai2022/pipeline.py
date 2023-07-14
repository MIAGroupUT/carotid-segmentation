from os import path, makedirs
import pandas as pd
from monai.data import ITKWriter
import xml.etree.ElementTree as ET
from carotid.utils import build_dataset, ContourSerializer, write_json
from carotid.utils.transforms import CropBackground
from carotid.convert.utils import find_annotated_slices, get_contour


def convert(
    original_dir: str,
    raw_dir: str,
    annotation_dir: str,
):
    dataset = build_dataset(raw_dir=original_dir)

    rescaling_parameters = {
        "rescale": True,
        "lower_percentile_rescaler": 0,
        "upper_percentile_rescaler": 100,
    }
    crop_transform = CropBackground()
    writer = ITKWriter()

    serializer = ContourSerializer(annotation_dir)
    columns = ["label", "object", "x", "y", "z"]

    makedirs(raw_dir, exist_ok=True)
    write_json(rescaling_parameters, path.join(raw_dir, "parameters.json"))

    for sample in dataset:
        participant_id = sample["participant_id"]
        participant_path = path.join(original_dir, participant_id)
        formatted_participant_id = f"sub-MICCAI2022P{participant_id}"
        sample["participant_id"] = formatted_participant_id

        # Crop original images
        image_pt = sample["image"]
        cropped_pt, (pad_ant, pad_post) = crop_transform(image_pt)

        writer.set_data_array(cropped_pt)
        writer.set_metadata(cropped_pt.__dict__)
        writer.write(
            path.join(raw_dir, f"{formatted_participant_id}.mha"), compression=True
        )

        spatial_dict = {
            "affine": image_pt.affine.tolist(),
            "orig_shape": cropped_pt[0].shape,
        }

        for side in ["left", "right"]:
            sample[f"{side}_contour"] = pd.DataFrame(columns=columns)
            sample[f"{side}_contour_meta_dict"] = spatial_dict

            qjv_path = path.join(
                participant_path, f"{participant_id}{side.upper()[0]}.QVJ"
            )
            if path.exists(qjv_path):
                qvj_root = ET.parse(qjv_path).getroot()
                qvs_filename = (
                    qvj_root.find("QVAS_Loaded_Series_List")
                    .find("QVASSeriesFileName")
                    .text
                )
                qvs_path = path.join(participant_path, qvs_filename)

                qvsroot = ET.parse(qvs_path).getroot()
                avail_slices = find_annotated_slices(qvsroot)

                for slice_idx in avail_slices:
                    try:
                        lumen_cont = get_contour(
                            qvsroot,
                            slice_idx,
                            "Lumen",
                            image_size=image_pt.shape[1],
                            check_integrity=False,
                        )
                        wall_cont = get_contour(
                            qvsroot,
                            slice_idx,
                            "Outer Wall",
                            image_size=image_pt.shape[1],
                            check_integrity=False,
                        )

                        # RAS convention
                        lumen_cont[:, 0] = image_pt.shape[-2] - lumen_cont[:, 0] - 1
                        lumen_cont[:, 1] = (
                            image_pt.shape[-1] - lumen_cont[:, 1] - pad_ant - 1
                        )

                        wall_cont[:, 0] = image_pt.shape[-2] - wall_cont[:, 0] - 1
                        wall_cont[:, 1] = (
                            image_pt.shape[-1] - wall_cont[:, 1] - pad_ant - 1
                        )

                        lumen_df = pd.DataFrame(lumen_cont, columns=["x", "y"])
                        lumen_df["object"] = "lumen"
                        wall_df = pd.DataFrame(wall_cont, columns=["x", "y"])
                        wall_df["object"] = "wall"
                        slice_df = pd.concat((lumen_df, wall_df))
                        slice_df["z"] = slice_idx
                        slice_df["label"] = "internal"

                        sample[f"{side}_contour"] = pd.concat(
                            (sample[f"{side}_contour"], slice_df)
                        )

                    except Exception:
                        print(
                            f"Participant {participant_id}, internal slice {slice_idx} could not be processed"
                        )

        if len(sample["left_contour"]) > 0 or len(sample["right_contour"]) > 0:
            serializer.write(sample)
