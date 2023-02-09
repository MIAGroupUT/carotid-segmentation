from os import path
import torch
from copy import deepcopy
from carotid.utils import build_dataset
from carotid.utils.transforms import ExtractLeftAndRightd, BuildEmptyHeatmapd

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def compare_left_and_right(original_sample, transformed_sample, reconstructed_sample):
    assert torch.equal(original_sample["image"], reconstructed_sample["image"])

    def extract_left(image_pt: torch.Tensor) -> torch.Tensor:
        """Extract the first two thirds of the image."""
        return torch.flip(image_pt[..., 0 : 2 * image_pt.shape[-1] // 3], dims=(-1,))

    def extract_right(image_pt: torch.Tensor) -> torch.Tensor:
        """Extract the last two thirds of the image and flip it."""
        return image_pt[..., image_pt.shape[-1] // 3 : :]

    for side, extract_fn in {"left": extract_left, "right": extract_right}.items():
        assert torch.equal(
            extract_fn(original_sample[f"{side}_heatmap"]),
            transformed_sample[f"{side}_heatmap"],
        )
        assert torch.equal(
            extract_fn(original_sample["image"]), transformed_sample[f"{side}_image"]
        )


def test_repro_extract_left_right_transform():
    raw_dir = path.join(test_dir, "raw_dir")
    heatmap_dir = path.join(test_dir, "heatmap_transform", "reference")

    transform = ExtractLeftAndRightd(split_keys=["heatmap"], keys=["image", "heatmap"])

    # Read reference
    dataset = build_dataset(heatmap_dir=heatmap_dir, raw_dir=raw_dir)

    for sample in dataset:
        transformed_sample = transform(deepcopy(sample))
        reconstructed_sample = transform.inverse(deepcopy(transformed_sample))
        compare_left_and_right(sample, transformed_sample, reconstructed_sample)
