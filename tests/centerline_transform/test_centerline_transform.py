from os import path
from carotid.utils import build_dataset
from carotid.centerline_transform.pipeline import apply_transform
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "heatmap_transform", "reference")
    ref_dir = path.join(test_dir, "centerline_transform", "reference")

    apply_transform(
        output_dir=tmp_dir,
        heatmap_dir=input_dir,
        config_path=path.join(test_dir, "centerline_transform", "test_args.toml"),
    )

    # Read reference
    ref_dataset = build_dataset(centerline_dir=ref_dir)

    # Read output
    out_dataset = build_dataset(centerline_dir=tmp_dir)

    for side in ["left", "right"]:
        ref_df = ref_dataset[0][f"{side}_centerline"]
        out_df = out_dataset[0][f"{side}_centerline"]
        assert ref_df.equals(out_df)

    shutil.rmtree(tmp_dir)
