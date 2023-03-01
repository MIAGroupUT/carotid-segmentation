from os import path
from carotid.utils import build_dataset, read_json, check_equal_parameters
from carotid.transform.segmentation.pipeline import apply_transform
import numpy as np
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "segmentation_transform", "input")
    ref_dir = path.join(test_dir, "segmentation_transform", "reference")

    apply_transform(
        output_dir=tmp_dir,
        contour_dir=input_dir,
        force=True,
    )

    # Read reference
    ref_dataset = build_dataset(segmentation_dir=ref_dir)

    # Read output
    out_dataset = build_dataset(segmentation_dir=tmp_dir)

    # Compare parameters
    ref_params = read_json(path.join(ref_dir, "parameters.json"))
    out_params = read_json(path.join(tmp_dir, "parameters.json"))
    check_equal_parameters(ref_params, out_params)

    for side in ["left", "right"]:
        ref_np = ref_dataset[0][f"{side}_segmentation"]
        out_np = out_dataset[0][f"{side}_segmentation"]
        assert np.all(ref_np == out_np)

    shutil.rmtree(tmp_dir)
