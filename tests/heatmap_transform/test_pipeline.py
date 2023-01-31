from os import path
import numpy as np
from carotid.utils import build_dataset, HeatmapLogger
from carotid.heatmap_transform.pipeline import apply_transform

import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_first_lv():

    apply_transform(
        raw_dir=path.join(test_dir, "raw_dir"),
        model_dir=path.join(test_dir, "models", "heatmap_transform"),
        output_dir=path.join(test_dir, "tmp"),
    )

    # Read reference
    ref_dataset = build_dataset(
        [HeatmapLogger({"dir": path.join(test_dir, "reference")})]
    )

    # Read output
    out_dataset = build_dataset([HeatmapLogger({"dir": path.join(test_dir, "tmp")})])

    ref_sample = ref_dataset[0]
    out_sample = out_dataset[0]

    assert np.all(ref_sample["left_heatmap"] == out_sample["left_heatmap"])
    assert np.all(ref_sample["right_heatmap"] == out_sample["right_heatmap"])
    shutil.rmtree(path.join(test_dir, "tmp"))
