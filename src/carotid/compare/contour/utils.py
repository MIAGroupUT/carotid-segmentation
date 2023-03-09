import pandas as pd
from shapely import Polygon
import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple


def compute_dice_scores(
    lumen1_np: np.ndarray, lumen2_np: np.ndarray, wall1_np: np.ndarray, wall2_np: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Dice scores for the lumen and wall contours.

    Returns:
        - dice score for the lumen,
        - dice score for the wall.
    """
    # Lumen score
    lumen1_poly = Polygon(lumen1_np).convex_hull
    lumen2_poly = Polygon(lumen2_np).convex_hull
    lumen_inter_poly = lumen1_poly.intersection(lumen2_poly)

    lumen_dice = 2 * lumen_inter_poly.area / (lumen1_poly.area + lumen2_poly.area)

    # Wall score
    wall1_poly = Polygon(wall1_np).convex_hull.difference(lumen1_poly)
    wall2_poly = Polygon(wall2_np).convex_hull.difference(lumen2_poly)
    wall_inter_poly = wall1_poly.intersection(wall2_poly)

    wall_dice = 2 * wall_inter_poly.area / (wall1_poly.area + wall2_poly.area)

    return lumen_dice, wall_dice


def compute_point_distance(
    transform_df: pd.DataFrame, in_reference_df: pd.DataFrame
) -> pd.DataFrame:

    output_df = transform_df.copy()
    if "label" not in output_df.columns:
        output_df.reset_index(inplace=True)
    output_df.set_index(["label", "object", "z"], inplace=True)
    output_df.sort_index(inplace=True)

    reference_df = in_reference_df.copy()
    if "label" not in reference_df.columns:
        reference_df.reset_index(inplace=True)
    reference_df.set_index(["label", "object", "z"], inplace=True)
    reference_df.sort_index(inplace=True)

    for label_name, object_name, slice_idx in output_df.index.unique():
        print(label_name, object_name, slice_idx)
        try:
            transform_contour_np = output_df.loc[(label_name, object_name, slice_idx), ["x", "y"]].values.astype(float)
            reference_contour_np = reference_df.loc[(label_name, object_name, slice_idx), ["x", "y"]].values.astype(float)
            print(transform_contour_np)
            min_distances = np.min(cdist(transform_contour_np, reference_contour_np), axis=1)
            output_df.loc[(label_name, object_name, slice_idx), "min_distance"] = min_distances

        except KeyError:
            output_df.pop((label_name, object_name, slice_idx))

    return output_df.reset_index()