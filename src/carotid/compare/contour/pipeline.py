from carotid.utils import build_dataset
from carotid.compare.contour.utils import compute_dice_scores, compute_point_distance
from os import path, makedirs
import pandas as pd


def compare(
    transform_dir: str,
    reference_dir: str,
    output_dir: str,
    dice_filename: str = "compare_contour_dice.tsv",
    point_filename: str = "compare_contour_points.tsv",
):
    transform_set = build_dataset(contour_dir=transform_dir)
    reference_set = build_dataset(contour_dir=reference_dir)
    makedirs(output_dir, exist_ok=True)

    # Find all relevant participant IDs
    dataset1_id_list = list()
    for sample in transform_set:
        dataset1_id_list.append(sample["participant_id"])

    dataset2_id_list = list()
    for sample in reference_set:
        dataset2_id_list.append(sample["participant_id"])

    common_id_set = set(dataset1_id_list) & set(dataset2_id_list)

    cols = ["participant_id", "side", "label", "object", "z", "dice_score"]
    dice_df = pd.DataFrame(columns=cols)
    point_df = pd.DataFrame()

    for participant_id in common_id_set:
        transform_sample = transform_set[dataset1_id_list.index(participant_id)]
        reference_sample = reference_set[dataset2_id_list.index(participant_id)]

        for side in ["left", "right"]:
            transform_contour_df = transform_sample[f"{side}_contour"]
            reference_contour_df = reference_sample[f"{side}_contour"]

            # Point by point comparison
            side_df = compute_point_distance(transform_contour_df, reference_contour_df)
            side_df["side"] = side
            side_df["participant_id"] = participant_id
            point_df = pd.concat((point_df, side_df))

            transform_contour_df.set_index(["label", "z"], inplace=True)
            transform_contour_df.sort_index(inplace=True)
            reference_contour_df.set_index(["label", "z"], inplace=True)
            reference_contour_df.sort_index(inplace=True)

            for label_name, slice_idx in transform_contour_df.index.unique():
                try:
                    transform_lumen_np = transform_contour_df[transform_contour_df.object == "lumen"].loc[(label_name, slice_idx), ["x", "y"]]
                    reference_lumen_np = reference_contour_df[reference_contour_df.object == "lumen"].loc[(label_name, slice_idx), ["x", "y"]]
                    transform_wall_np = transform_contour_df[transform_contour_df.object == "wall"].loc[(label_name, slice_idx), ["x", "y"]]
                    reference_wall_np = reference_contour_df[reference_contour_df.object == "wall"].loc[(label_name, slice_idx), ["x", "y"]]
                    lumen_score, wall_score = compute_dice_scores(
                        transform_lumen_np, reference_lumen_np, transform_wall_np, reference_wall_np
                    )
                    row_df = pd.DataFrame(
                        [
                            [participant_id, side, label_name, "lumen", slice_idx, lumen_score],
                            [participant_id, side, label_name, "wall", slice_idx, wall_score]
                        ],
                        columns=cols
                    )
                    dice_df = pd.concat((dice_df, row_df))

                except KeyError:
                    pass

    dice_df.set_index(["participant_id", "side", "label", "z", "object"], inplace=True)
    dice_df.sort_index(inplace=True)
    dice_df.to_csv(path.join(output_dir, dice_filename), sep="\t")
    point_df.set_index(["participant_id", "side", "label", "z", "object"], inplace=True)
    point_df.sort_index(inplace=True)
    point_df.to_csv(path.join(output_dir, point_filename), sep="\t")
