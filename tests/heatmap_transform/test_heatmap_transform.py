from os import path
import torch
from carotid.utils import build_dataset
from carotid.heatmap_transform.pipeline import apply_transform

import shutil

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def test_pipeline():
    tmp_dir = path.join(test_dir, "tmp")
    ref_dir = path.join(test_dir, "heatmap_transform", "reference")

    apply_transform(
        raw_dir=path.join(test_dir, "raw_dir"),
        model_dir=path.join(test_dir, "models", "heatmap_transform"),
        config_path=path.join(test_dir, "heatmap_transform", "test_args.toml"),
        output_dir=tmp_dir,
    )

    # Read reference
    ref_dataset = build_dataset(heatmap_dir=ref_dir)

    # Read output
    out_dataset = build_dataset(heatmap_dir=tmp_dir)

    ref_sample = ref_dataset[0]
    out_sample = out_dataset[0]

    for side in ["left", "right"]:
        assert (
            torch.max(torch.abs(ref_sample[f"{side}_heatmap"] - out_sample[f"{side}_heatmap"]))
            < 1e-3
        )
    shutil.rmtree(path.join(test_dir, "tmp"))
