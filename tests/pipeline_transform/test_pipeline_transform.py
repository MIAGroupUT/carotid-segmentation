from os import path
import numpy as np
from carotid.utils import build_dataset
from carotid.pipeline_transform.pipeline import apply_transform

import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    ref_dir = path.join(test_dir, "pipeline_transform", "reference")

    apply_transform(
        raw_dir=path.join(test_dir, "raw_dir"),
        heatmap_model_dir=path.join(test_dir, "models", "heatmap_transform"),
        contour_model_dir=path.join(test_dir, "models", "contour_transform"),
        output_dir=tmp_dir,
    )

    # Read reference
    ref_dataset = build_dataset(contour_dir=ref_dir, segmentation_dir=ref_dir)

    # Read output
    out_dataset = build_dataset(contour_dir=tmp_dir, segmentation_dir=tmp_dir)

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

    shutil.rmtree(path.join(test_dir, "tmp"))
