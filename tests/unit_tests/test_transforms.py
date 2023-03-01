from os import path
import torch
from copy import deepcopy
from carotid.utils import build_dataset
from carotid.utils.transforms import ExtractLeftAndRightd, polar2cart, cart2polar, Printd

test_dir = path.dirname(path.dirname(path.realpath(__file__)))


def compare_left_and_right(original_sample, transformed_sample, reconstructed_sample):
    assert torch.equal(original_sample["image"], reconstructed_sample["image"])

    def extract_left(image_pt: torch.Tensor) -> torch.Tensor:
        """Extract the first two thirds of the image."""
        return image_pt[..., image_pt.shape[-1] // 3 : :]

    def extract_right(image_pt: torch.Tensor) -> torch.Tensor:
        """Extract the last two thirds of the image and flip it."""
        return torch.flip(image_pt[..., 0 : 2 * image_pt.shape[-1] // 3], dims=(-1,))

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
    heatmap_dir = path.join(test_dir, "transform", "heatmap", "reference")

    transform = ExtractLeftAndRightd(split_keys=["heatmap"], keys=["image", "heatmap"])

    # Read reference
    dataset = build_dataset(heatmap_dir=heatmap_dir, raw_dir=raw_dir)

    for sample in dataset:
        for side in ["left", "right"]:
            sample[f"{side}_heatmap"] = sample[f"{side}_heatmap"]["mean"]
        transformed_sample = transform(deepcopy(sample))
        reconstructed_sample = transform.inverse(deepcopy(transformed_sample))
        compare_left_and_right(sample, transformed_sample, reconstructed_sample)


def test_cart2polar():
    dataset = build_dataset(
        contour_dir=path.join(test_dir, "transform", "contour",  "reference")
    )

    sample = dataset[0]
    for side in ["left", "right"]:
        contour_df = sample[f"{side}_contour"]
        obj_df = contour_df[
            (contour_df.label == "internal") & (contour_df.object == "lumen")
        ]
        slice_idx = obj_df.z.median()
        slice_df = obj_df[obj_df.z == slice_idx]
        cart_pt = torch.from_numpy(slice_df[["z", "y", "x"]].values)
        center_pt = torch.mean(cart_pt, dim=0)
        polar_pt = cart2polar(cart_pt, center_pt)
        inverse_pt = polar2cart(polar_pt, center_pt)

        assert torch.allclose(cart_pt, inverse_pt)


def test_print():
    raw_dir = path.join(test_dir, "raw_dir")
    dataset = build_dataset(raw_dir=raw_dir)
    transform = Printd(keys=["image"], message="test")

    for sample in dataset:
        transformed_sample = transform(deepcopy(sample))
        reconstructed_sample = transform.inverse(deepcopy(transformed_sample))
        assert torch.equal(sample["image"], transformed_sample["image"])
        assert torch.equal(sample["image"], reconstructed_sample["image"])
