from carotid.transforms.centerline.utils import CenterlineExtractor
import pandas as pd


def test_remove_common():

    input_df = pd.DataFrame(
        [
            ["internal", 1, 1, 3],
            ["internal", 2, 1, 3],
            ["internal", 3, 1, 3],
            ["internal", 4, 1, 3],
            ["internal", 5, 1, 3],
            ["external", 0, 1, 3],
            ["external", 1, 1, 3],
            ["external", 2, 1, 3],
            ["external", 3, 2, 3],
            ["external", 4, 1, 3],
            ["external", 5, 1, 3],
            ["external", 6, 1, 3],
        ],
        columns=["label", "z", "x", "y"]
    )

    ref_df = pd.DataFrame(
        [
            ["internal", 0, 1, 3],
            ["internal", 1, 1, 3],
            ["internal", 2, 1, 3],
            ["internal", 3, 1, 3],
            ["internal", 4, 1, 3],
            ["internal", 5, 1, 3],
            ["external", 3, 2, 3],
            ["external", 4, 1, 3],
            ["external", 5, 1, 3],
            ["external", 6, 1, 3],
        ],
        columns=["label", "z", "x", "y"]
    )

    centerline_extractor = CenterlineExtractor({"spatial_threshold": 0})
    out_df = centerline_extractor.remove_common_external_centers(input_df)

    ref_df.set_index(["label", "z"], inplace=True)
    ref_df.sort_index(inplace=True)

    out_df.set_index(["label", "z"], inplace=True)
    out_df.sort_index(inplace=True)

    assert ref_df.equals(out_df)
