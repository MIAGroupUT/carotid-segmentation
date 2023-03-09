from os import path, remove
import pandas as pd
from carotid.compare.centerline.pipeline import compare

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_tsv = path.join(test_dir, "tmp.tsv")
    input_dir = path.join(test_dir, "centerline", "input")
    ref_tsv = path.join(test_dir, "centerline", "reference.tsv")

    compare(
        transform1_dir=path.join(input_dir, "transform1"),
        transform2_dir=path.join(input_dir, "transform2"),
        output_path=tmp_tsv,
    )

    # Read reference
    ref_df = pd.read_csv(ref_tsv, sep="\t")

    # Read output
    out_df = pd.read_csv(tmp_tsv, sep="\t")

    assert ref_df.equals(out_df)

    remove(tmp_tsv)
