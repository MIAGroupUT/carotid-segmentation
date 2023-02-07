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
        segmentation_model_dir=path.join(test_dir, "models", "segmentation_transform"),
        output_dir=tmp_dir,
    )

    # Read reference
    ref_dataset = build_dataset(segmentation_dir=ref_dir)

    # Read output
    out_dataset = build_dataset(segmentation_dir=tmp_dir)

    for side in ["left", "right"]:
        ref_np = ref_dataset[0][f"{side}_segmentation"]
        out_np = out_dataset[0][f"{side}_segmentation"]
        assert np.allclose(ref_np, out_np)

    shutil.rmtree(path.join(test_dir, "tmp"))
