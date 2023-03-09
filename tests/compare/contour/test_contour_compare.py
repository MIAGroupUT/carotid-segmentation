from os import path
import pandas as pd
from carotid.compare.contour.pipeline import compare
import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_path = path.join(test_dir, "tmp")
    input_dir = path.join(test_dir, "contour", "input")
    ref_path = path.join(test_dir, "contour", "reference")

    compare(
        transform_dir=path.join(input_dir, "transform"),
        reference_dir=path.join(input_dir, "reference"),
        output_dir=tmp_path,
    )

    for filename in ["compare_contour_dice.tsv", "compare_contour_points.tsv"]:
        ref_df = pd.read_csv(path.join(ref_path, filename), sep="\t")
        out_df = pd.read_csv(path.join(tmp_path, filename), sep="\t")
        assert ref_df.equals(out_df)

    shutil.rmtree(tmp_path)
