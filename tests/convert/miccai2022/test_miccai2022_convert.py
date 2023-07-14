from os import path
from carotid.utils import build_dataset
from carotid.convert.miccai2022.pipeline import convert
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_raw_dir = path.join(test_dir, "tmp", "raw")
    tmp_annotation_dir = path.join(test_dir, "tmp", "annotation")
    input_dir = path.join(test_dir, "miccai2022", "input")
    ref_raw_dir = path.join(test_dir, "miccai2022", "reference", "raw")
    ref_annotation_dir = path.join(test_dir, "miccai2022", "reference", "annotation")

    convert(
        original_dir=input_dir,
        raw_dir=tmp_raw_dir,
        annotation_dir=tmp_annotation_dir,
    )

    # Read reference
    ref_dataset = build_dataset(
        raw_dir=ref_raw_dir,
        contour_dir=ref_annotation_dir,
    )

    # Read output
    out_dataset = build_dataset(
        raw_dir=tmp_raw_dir,
        contour_dir=tmp_annotation_dir,
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

    shutil.rmtree(path.join(test_dir, "tmp"))
