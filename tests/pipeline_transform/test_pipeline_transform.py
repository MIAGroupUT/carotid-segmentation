from os import path
import numpy as np
from carotid.utils import build_dataset, read_json, check_equal_parameters
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
        config_path=path.join(test_dir, "pipeline_transform", "test_args.toml"),
        force=True,
    )

    # Read reference
    ref_dataset = build_dataset(contour_dir=ref_dir)

    # Read output
    out_dataset = build_dataset(contour_dir=tmp_dir)

    # Compare parameters
    ref_params = read_json(path.join(ref_dir, "parameters.json"))
    out_params = read_json(path.join(tmp_dir, "parameters.json"))
    check_equal_parameters(ref_params, out_params)

    # Compare parameters
    ref_params = read_json(path.join(ref_dir, "parameters.json"))
    out_params = read_json(path.join(tmp_dir, "parameters.json"))
    check_equal_parameters(ref_params, out_params)

    # Only compare contours
    # for side in ["left", "right"]:
    #     ref_df = ref_dataset[0][f"{side}_contour"].set_index(
    #         ["label", "object", "z"], drop=True
    #     )
    #     ref_df.sort_index(inplace=True)
    #     out_df = out_dataset[0][f"{side}_contour"].set_index(
    #         ["label", "object", "z"], drop=True
    #     )
    #     out_df.sort_index(inplace=True)
    #     for index, ref_slice_df in ref_df.groupby(["label", "object", "z"]):
    #         out_slice_df = out_df.loc[index]
    #         out_slice_np = out_slice_df.values
    #         ref_slice_np = ref_slice_df.values
    #         print(np.max(np.abs(out_slice_np - ref_slice_np)))
    #         assert np.allclose(ref_slice_np, out_slice_np, rtol=1e-3, atol=0.1)

    shutil.rmtree(path.join(test_dir, "tmp"))
