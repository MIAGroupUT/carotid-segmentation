from os import path
import numpy as np
from carotid.utils import build_dataset, PolarLogger
from carotid.polar_transform.pipeline import apply_transform
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    raw_dir = path.join(test_dir, "raw_dir")
    input_dir = path.join(test_dir, "centerline_transform", "reference")
    ref_dir = path.join(test_dir, "polar_transform", "reference")

    shutil.copytree(input_dir, tmp_dir)

    apply_transform(
        raw_dir=raw_dir,
        output_dir=tmp_dir,
    )

    # Read reference
    ref_dataset = build_dataset([PolarLogger({"dir": ref_dir})])

    # Read output
    out_dataset = build_dataset([PolarLogger({"dir": tmp_dir})])

    for side in ["left", "right"]:
        ref_list = ref_dataset[0][f"{side}_polar"]
        out_list = out_dataset[0][f"{side}_polar"]
        assert len(ref_list) == len(out_list)
        for idx in range(len(ref_list)):
            assert np.allclose(ref_list[idx]["polar_img"], out_list[idx]["polar_img"])

    shutil.rmtree(tmp_dir)
