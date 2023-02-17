from os import path
from carotid.utils import build_dataset
from carotid.contour_transform.pipeline import apply_transform
import numpy as np
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "polar_transform", "reference")
    ref_dir = path.join(test_dir, "contour_transform", "reference")
    model_dir = path.join(test_dir, "models", "contour_transform")

    apply_transform(
        output_dir=tmp_dir,
        polar_dir=input_dir,
        model_dir=model_dir,
    )

    # Read reference
    ref_dataset = build_dataset(contour_dir=ref_dir)

    # Read output
    out_dataset = build_dataset(contour_dir=tmp_dir)

    for side in ["left", "right"]:
        ref_df = ref_dataset[0][f"{side}_contour"].set_index(
            ["label", "object", "z"], drop=True
        )
        out_df = out_dataset[0][f"{side}_contour"].set_index(
            ["label", "object", "z"], drop=True
        )
        for index, ref_slice_df in ref_df.groupby(["label", "object", "z"]):
            out_slice_df = out_df.loc[index]
            out_slice_np = out_slice_df.values
            ref_slice_np = ref_slice_df.values
            assert np.allclose(ref_slice_np, out_slice_np, rtol=1e-3, atol=1e-5)

    shutil.rmtree(tmp_dir)


def test_pipeline_dropout():
    tmp_dir = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "polar_transform", "reference")
    model_dir = path.join(test_dir, "models", "contour_transform_dropout")
    config_path = path.join(test_dir, "contour_transform", "test_args_dropout.toml")

    apply_transform(
        output_dir=tmp_dir,
        polar_dir=input_dir,
        model_dir=model_dir,
        config_path=config_path,
    )

    shutil.rmtree(tmp_dir)