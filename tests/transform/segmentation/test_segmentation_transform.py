from os import path
from carotid.utils import build_dataset, read_json, check_equal_parameters
from carotid.transform.segmentation.pipeline import apply_transform
import numpy as np
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "segmentation", "input")
    ref_dir = path.join(test_dir, "segmentation", "reference")

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

        # Lumen
        out_lumen_np = (out_np[0] + out_np[2]).clip(0, 1)
        assert np.all(ref_np[0] == out_lumen_np)

        # Wall comparison
        out_wall_np = (out_np[1] + out_np[3]).clip(0, 1)
        assert np.all(ref_np[0] + ref_np[1] == out_wall_np)

    shutil.rmtree(tmp_dir)
