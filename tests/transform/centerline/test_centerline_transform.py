from os import path
from carotid.utils import build_dataset, read_json, check_equal_parameters
from carotid.transform.centerline.pipeline import apply_transform
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "centerline", "input")
    ref_dir = path.join(test_dir, "centerline", "reference")

    apply_transform(
        output_dir=tmp_dir,
        heatmap_dir=input_dir,
        config_path=path.join(test_dir, "centerline", "test_args.toml"),
        force=True,
    )

    # Read reference
    ref_dataset = build_dataset(centerline_dir=ref_dir)

    # Read output
    out_dataset = build_dataset(centerline_dir=tmp_dir)

    # Compare parameters
    ref_params = read_json(path.join(ref_dir, "parameters.json"))
    out_params = read_json(path.join(tmp_dir, "parameters.json"))
    check_equal_parameters(ref_params, out_params)

    for side in ["left", "right"]:
        ref_df = ref_dataset[0][f"{side}_centerline"][["label", "x", "y", "z"]]
        out_df = out_dataset[0][f"{side}_centerline"][["label", "x", "y", "z"]]
        assert ref_df.equals(out_df)

    shutil.rmtree(tmp_dir)
