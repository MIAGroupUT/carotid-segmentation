from os import path
from carotid.utils import build_dataset
from carotid.convert.miccai2020.pipeline import convert
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "miccai2020", "input")
    ref_dir = path.join(test_dir, "miccai2020", "reference")

    convert(
        original_dir=input_dir,
        output_dir=tmp_dir,
    )

    # Read reference
    ref_dataset = build_dataset(
        raw_dir=ref_dir,
        contour_dir=ref_dir,
    )

    # Read output
    out_dataset = build_dataset(
        raw_dir=tmp_dir,
        contour_dir=tmp_dir,
    )

    ref_sample = ref_dataset[0]
    out_sample = out_dataset[0]

    assert (ref_sample["image"] == out_sample["image"]).all()

    for side in ["left", "right"]:
        ref_df = ref_sample[f"{side}_contour"].set_index(
            ["label", "object", "z"], drop=True
        )
        ref_df.sort_index(inplace=True)
        out_df = out_sample[f"{side}_contour"].set_index(
            ["label", "object", "z"], drop=True
        )
        out_df.sort_index(inplace=True)
        for index, ref_slice_df in ref_df.groupby(["label", "object", "z"]):
            out_slice_df = out_df.loc[index]
            out_slice_np = out_slice_df[["x", "y"]].values
            ref_slice_np = ref_slice_df[["x", "y"]].values
            assert (ref_slice_np == out_slice_np).all()

    shutil.rmtree(tmp_dir)
