from os import path
import pandas as pd
from carotid.utils import read_json, check_equal_parameters
from carotid.transform.pipeline.pipeline import apply_transform
from carotid.compare.contour.pipeline import compare

import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    ref_dir = path.join(test_dir, "pipeline", "reference")

    apply_transform(
        raw_dir=path.join(test_dir, "..", "raw_dir"),
        heatmap_model_dir=path.join(test_dir, "..", "models", "heatmap_transform"),
        contour_model_dir=path.join(test_dir, "..", "models", "contour_transform"),
        output_dir=tmp_dir,
        config_path=path.join(test_dir, "pipeline", "test_args.toml"),
        force=True,
        write_contours=True,
    )

    compare(
        transform_dir=tmp_dir,
        reference_dir=ref_dir,
        output_dir=tmp_dir,
    )

    # Compare parameters
    ref_params = read_json(path.join(ref_dir, "parameters.json"))
    out_params = read_json(path.join(tmp_dir, "parameters.json"))
    check_equal_parameters(ref_params, out_params)

    # Compare parameters
    ref_params = read_json(path.join(ref_dir, "parameters.json"))
    out_params = read_json(path.join(tmp_dir, "parameters.json"))
    check_equal_parameters(ref_params, out_params)

    dice_df = pd.read_csv(path.join(tmp_dir, "compare_contour_dice.tsv"), sep="\t")
    print(dice_df)
    assert (dice_df.dice_score > 0.99).all()

    shutil.rmtree(tmp_dir)
