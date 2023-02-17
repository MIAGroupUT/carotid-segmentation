from os import path
import torch
from carotid.utils import build_dataset
from carotid.polar_transform.pipeline import apply_transform
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "centerline_transform", "reference")
    ref_dir = path.join(test_dir, "polar_transform", "reference")

    apply_transform(
        output_dir=tmp_dir,
        centerline_dir=input_dir,
    )

    # Read reference
    ref_dataset = build_dataset(polar_dir=ref_dir)

    # Read output
    out_dataset = build_dataset(polar_dir=tmp_dir)

    for side in ["left", "right"]:
        ref_list = ref_dataset[0][f"{side}_polar"]
        out_list = out_dataset[0][f"{side}_polar"]
        assert len(ref_list) == len(out_list)
        for idx in range(len(ref_list)):
            assert torch.allclose(
                ref_list[idx]["polar_pt"],
                out_list[idx]["polar_pt"],
                rtol=1e-4,
                atol=1e-5,
            )

    shutil.rmtree(tmp_dir)
