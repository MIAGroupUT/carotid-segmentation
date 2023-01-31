from os import path
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
    # ref_dataset = build_dataset([PolarLogger({"dir": ref_dir})])

    # Read output
    # out_dataset = build_dataset([PolarLogger({"dir": tmp_dir})])

    # for side in ["left", "right"]:
    #     ref_df = ref_dataset[0][f"{side}_centerline"]
    #     out_df = out_dataset[0][f"{side}_centerline"]
    #     assert ref_df.equals(out_df)

    # shutil.rmtree(tmp_dir)
