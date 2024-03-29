from os import path, makedirs
from shutil import copy
from glob import glob
import pandas as pd
import xml.etree.ElementTree as ET
from carotid.convert.utils import get_contour, find_annotated_slices
from carotid.utils import build_dataset, ContourSerializer, write_json
from logging import getLogger

logger = getLogger("carotid")


def convert(
    original_dir: str,
    raw_dir: str,
    annotation_dir: str,
):

    dataset = build_dataset(raw_dir=original_dir)
    rescaling_parameters = {
        "rescale": True,
        "lower_percentile_rescaler": 5,
        "upper_percentile_rescaler": 95,
    }

    keys_dict = {
        "ICAL": {"label": "internal", "side": "left"},
        "ECAL": {"label": "external", "side": "left"},
        "ICAR": {"label": "internal", "side": "right"},
        "ECAR": {"label": "external", "side": "right"},
    }

    serializer = ContourSerializer(annotation_dir)
    columns = ["label", "object", "x", "y", "z"]
    makedirs(raw_dir, exist_ok=True)
    write_json(rescaling_parameters, path.join(raw_dir, "parameters.json"))

    for sample in dataset:
        participant_id = sample["participant_id"]
        cascade_participant_id = participant_id.split("_")[1]
        formatted_participant_id = f"sub-MICCAI2020{cascade_participant_id}"
        sample["participant_id"] = formatted_participant_id
        participant_path = path.join(original_dir, participant_id)

        makedirs(path.join(raw_dir, formatted_participant_id))
        dcm_paths = glob(path.join(participant_path, "*.dcm"))
        for dcm_path in dcm_paths:
            copy(dcm_path, path.join(raw_dir, formatted_participant_id))

        image_pt = sample["image"]
        spatial_dict = {
            "affine": image_pt.affine.tolist(),
            "orig_shape": image_pt[0].shape,
        }
        slice_pad_idx = (image_pt.shape[1] - image_pt.shape[2]) // 2

        for side in ["left", "right"]:
            sample[f"{side}_contour"] = pd.DataFrame(columns=columns)
            sample[f"{side}_contour_meta_dict"] = spatial_dict

        for vessel_type, vessel_dict in keys_dict.items():
            side = vessel_dict["side"]
            qvs_path = path.join(
                participant_path,
                f"CASCADE-{vessel_type}",
                f"E{cascade_participant_id}S101_L.QVS",
            )
            if path.exists(qvs_path):
                qvsroot = ET.parse(qvs_path).getroot()
                avail_slices = find_annotated_slices(qvsroot)

                for slice_idx in avail_slices:
                    try:
                        lumen_cont = get_contour(
                            qvsroot, slice_idx, "Lumen", image_size=image_pt.shape[1]
                        )
                        wall_cont = get_contour(
                            qvsroot,
                            slice_idx,
                            "Outer Wall",
                            image_size=image_pt.shape[1],
                        )
                        lumen_cont[:, 1], wall_cont[:, 1] = (
                            lumen_cont[:, 1] - slice_pad_idx,
                            wall_cont[:, 1] - slice_pad_idx,
                        )

                        # RAS convention
                        lumen_cont[:, 0] = image_pt.shape[-1] - lumen_cont[:, 0]
                        lumen_cont[:, 1] = image_pt.shape[-2] - lumen_cont[:, 1]

                        wall_cont[:, 0] = image_pt.shape[-1] - wall_cont[:, 0]
                        wall_cont[:, 1] = image_pt.shape[-2] - wall_cont[:, 1]

                        lumen_df = pd.DataFrame(lumen_cont, columns=["x", "y"])
                        lumen_df["object"] = "lumen"
                        wall_df = pd.DataFrame(wall_cont, columns=["x", "y"])
                        wall_df["object"] = "wall"
                        slice_df = pd.concat((lumen_df, wall_df))
                        slice_df["z"] = slice_idx
                        slice_df["label"] = vessel_dict["label"]

                        sample[f"{side}_contour"] = pd.concat(
                            (sample[f"{side}_contour"], slice_df)
                        )

                    except Exception:
                        logger.warning(
                            f"Participant {participant_id}, {vessel_type} slice {slice_idx} could not be processed"
                        )

            if len(sample["left_contour"]) > 0 and len(sample["right_contour"]) > 0:
                serializer.write(sample)
