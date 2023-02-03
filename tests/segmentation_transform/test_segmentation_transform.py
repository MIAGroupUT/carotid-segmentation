from os import path
from carotid.utils import build_dataset
from carotid.segmentation_transform.pipeline import apply_transform
import numpy as np
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "polar_transform", "reference")
    ref_dir = path.join(test_dir, "segmentation_transform", "reference")
    model_dir = path.join(test_dir, "models", "segmentation_transform")

    apply_transform(
        output_dir=tmp_dir,
        polar_dir=input_dir,
        model_dir=model_dir,
    )

    # Read reference
    ref_dataset = build_dataset(segmentation_parameters={"dir": ref_dir})

    # Read output
    out_dataset = build_dataset(segmentation_parameters={"dir": tmp_dir})

    for side in ["left", "right"]:
        ref_np = ref_dataset[0][f"{side}_segmentation"]
        out_np = out_dataset[0][f"{side}_segmentation"]
        assert np.allclose(ref_np, out_np)

    shutil.rmtree(tmp_dir)
