from os import path
import torch
from carotid.utils import build_dataset, read_json, check_equal_parameters
from carotid.transform.polar.pipeline import apply_transform
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline(test_similarity=False):
    tmp_dir = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "polar", "input")
    ref_dir = path.join(test_dir, "polar", "reference")

    apply_transform(
        output_dir=tmp_dir,
        centerline_dir=input_dir,
        config_path=path.join(test_dir, "polar", "test_args.toml"),
        force=True,
    )

    # Read reference
    ref_dataset = build_dataset(polar_dir=ref_dir)

    # Read output
    out_dataset = build_dataset(polar_dir=tmp_dir)

    # Compare parameters
    ref_params = read_json(path.join(ref_dir, "parameters.json"))
    out_params = read_json(path.join(tmp_dir, "parameters.json"))
    check_equal_parameters(ref_params, out_params)

    for side in ["left", "right"]:
        ref_list = ref_dataset[0][f"{side}_polar"]
        out_list = out_dataset[0][f"{side}_polar"]
        assert len(ref_list) == len(out_list)

        if test_similarity:
            for idx in range(len(ref_list)):
                print(ref_list[idx]["label"], out_list[idx]["label"])
                print(ref_list[idx]["slice_idx"], out_list[idx]["slice_idx"])
                print(
                    torch.max(
                        torch.abs(ref_list[idx]["polar_pt"] - out_list[idx]["polar_pt"])
                    )
                )
                assert torch.allclose(
                    ref_list[idx]["polar_pt"],
                    out_list[idx]["polar_pt"],
                    rtol=1e-4,
                    atol=1e-5,
                )

    shutil.rmtree(tmp_dir)
