from carotid.utils import build_dataset
from os import path
import numpy as np
import pandas as pd
from logging import getLogger


logger = getLogger("carotid")


def compare(transform1_dir: str, transform2_dir: str, output_path: str):
    dataset1 = build_dataset(centerline_dir=transform1_dir)
    dataset2 = build_dataset(centerline_dir=transform2_dir)

    if path.isdir(output_path):
        output_path = path.join(output_path, "compare_centerline.tsv")
    elif not output_path.endswith(".tsv"):
        output_path = f"{output_path}.tsv"

    # Find all relevant participant IDs
    dataset1_id_list = list()
    for sample in dataset1:
        dataset1_id_list.append(sample["participant_id"])

    dataset2_id_list = list()
    for sample in dataset2:
        dataset2_id_list.append(sample["participant_id"])

    common_id_set = set(dataset1_id_list) & set(dataset2_id_list)

    cols = ["participant_id", "side", "label", "z", "euclidean_distance"]
    output_df = pd.DataFrame(columns=cols)

    for participant_id in common_id_set:
        logger.info(f"Compare centerlines of {participant_id}")
        sample1 = dataset1[dataset1_id_list.index(participant_id)]
        sample2 = dataset2[dataset2_id_list.index(participant_id)]

        for side in ["left", "right"]:
            centerline1_df = sample1[f"{side}_centerline"]
            centerline2_df = sample2[f"{side}_centerline"]

            centerline1_df.set_index(["label", "z"], inplace=True)
            centerline1_df.sort_index(inplace=True)
            centerline2_df.set_index(["label", "z"], inplace=True)
            centerline2_df.sort_index(inplace=True)

            for label_name, slice_idx in centerline1_df.index.values:
                try:
                    center1 = centerline1_df.loc[(label_name, slice_idx), ["x", "y"]]
                    center2 = centerline2_df.loc[(label_name, slice_idx), ["x", "y"]]
                    distance = np.linalg.norm(center1 - center2)
                    row_df = pd.DataFrame(
                        [[participant_id, side, label_name, slice_idx, distance]],
                        columns=cols,
                    )
                    output_df = pd.concat((output_df, row_df))

                except KeyError:
                    pass

    output_df.to_csv(output_path, sep="\t", index=False)
