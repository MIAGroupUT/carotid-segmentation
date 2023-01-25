import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from os import path, listdir
from typing import Tuple, Dict, Any
import SimpleITK as sitk
from carotid.utils.transforms import ExtractLeftAndRightd, BuildEmptyLabelsd
from .post_processing import CenterlineExtractor, MarchingExtractor, OnePassExtractor

from monai.transforms import Flipd, Spacingd, Compose, InvertibleTransform, ToTensord
from monai.networks.nets import UNet
from monai.inferers import SlidingWindowInferer

side_list = ["left", "right"]


class UNetPredictor:
    """
    Transform MRI, compute the heatmap, and process it back to the original space.

    Args:
        model_dir: folder in which the weights of the pre-trained models are stored
        roi_size: size of the ROI used for inference
        flip_z: whether the images should be flipped to be correctly oriented before being fed to the U-Net.
        spacing: whether the images should be resampled to the U-Net pixdim.
        device: device used for computations.
    """

    def __init__(
        self,
        model_dir: str,
        roi_size: Tuple[int, int, int],
        flip_z: bool,
        spacing: bool,
        device: str = "cuda",
    ):

        # Remember all args
        self.model_dir = model_dir
        self.flip_z = flip_z
        self.spacing = spacing
        self.device = device

        # Create useful objects
        self.model_paths_list = [
            path.join(model_dir, filename)
            for filename in listdir(model_dir)
            if filename.endswith(".pt")
        ]
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(8, 16, 32, 64, 128),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.1,
        ).to(self.device)

        self.transforms = self.get_transforms(flip_z=flip_z, spacing=spacing)
        self.inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.8)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        # Remember original image to avoid a loss of resolution
        image_np = sample["img"].copy()
        # Put the sample in the working space of the U-Net
        unet_sample = self.transforms(sample)
        pred_dict = {
            side: torch.zeros_like(unet_sample[f"{side}_label"]) for side in side_list
        }

        # Compute sum of heatmaps for both sides
        for index, model_path in tqdm(
            enumerate(self.model_paths_list), desc="Predicting heatmaps", leave=False
        ):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

            for side in side_list:
                with torch.no_grad():
                    pred_dict[side] += (
                        self.inferer(
                            unet_sample[f"{side}_img"].unsqueeze(0).to(self.device),
                            self.model,
                        )
                        .squeeze(0)
                        .cpu()
                    )

        # Replace current label by model output
        for side in side_list:
            unet_sample[f"{side}_label"] = pred_dict[side] / len(self.model_paths_list)

        orig_sample = self.transforms.inverse(unet_sample)
        orig_sample["img"] = image_np

        return orig_sample

    @staticmethod
    def get_transforms(
        flip_z: bool = False,  # TODO: remove that
        spacing: bool = False,
    ) -> InvertibleTransform:
        """
        Outputs a series of invertible transforms to go from the original space to the
        working space of the U-Net.
        """

        key_list = ["img", "left_label", "right_label"]
        transforms = [BuildEmptyLabelsd(image_key="img")]
        if flip_z:
            transforms.append(Flipd(keys=key_list, spatial_axis=0))
        if spacing:
            transforms.append(Spacingd(keys=key_list, pixdim=[0.5, 0.5, 0.5]))

        transforms.append(
            ExtractLeftAndRightd(split_keys=["label"], keys=["img", "label"])
        )
        transforms.append(
            ToTensord(keys=["left_img", "right_img", "left_label", "right_label"])
        )
        # TODO: add transpose

        return Compose(transforms)


def get_centerline_extractor(method: str, **kwargs) -> CenterlineExtractor:
    if method == "one_pass":
        centerline_extractor = OnePassExtractor(**kwargs)
    elif method == "marching":
        centerline_extractor = MarchingExtractor(**kwargs)
    else:
        raise NotImplementedError(
            f"Method {method} for centerline extractor do not exist."
        )

    return centerline_extractor


def save_mevislab_markerfile(
    centerline_dict: Dict[str, Dict[str, np.ndarray]], output_dir: str
):

    for side in side_list:
        for label in ["internal", "external"]:
            output_path = path.join(output_dir, f"{side}_{label}_markers.xml")
            markers = centerline_dict[side][label]
            f = open(output_path, "w")
            print(r'<?xml version="1.0" encoding="UTF-8" standalone="no" ?>', file=f)
            print(r"<MeVis-XML-Data-v1.0>" + "\n", file=f)
            print(r'  <XMarkerList BaseType="XMarkerList" Version="1">', file=f)
            print(r'    <_ListBase Version="0">', file=f)
            print(r"      <hasPersistance>1</hasPersistance>", file=f)
            print(r"      <currentIndex>9</currentIndex>", file=f)
            print(r"    </_ListBase>", file=f)
            print(r"    <ListSize>" + str(markers.shape[0]) + "</ListSize>", file=f)
            print(r"    <ListItems>", file=f)
            for pt in range(markers.shape[0]):
                print(r'      <Item Version="0">', file=f)
                print(r'        <_BaseItem Version="0">', file=f)
                print(r"          <id>" + str(pt + 1) + "</id>", file=f)
                print(r"        </_BaseItem>", file=f)
                print(
                    r"        <pos>"
                    + str(markers[pt][0])
                    + " "
                    + str(markers[pt][1])
                    + " "
                    + str(markers[pt][2])
                    + r" 0 0 0</pos>",
                    file=f,
                )
                print(r"      </Item>", file=f)
            print(r"    </ListItems>", file=f)
            print(r"  </XMarkerList>" + "\n", file=f)
            print(r"</MeVis-XML-Data-v1.0>", file=f)


def save_heatmaps(sample: Dict[str, Any], output_dir: str):
    label_idx_dict = {"internal": 0, "external": 1}

    for side in side_list:
        for label in ["internal", "external"]:
            img_sitk = sitk.GetImageFromArray(
                sample[f"{side}_label"][label_idx_dict[label]]
            )
            img_sitk.SetSpacing(sample[f"{side}_label_meta_dict"]["spacing"])
            sitk.WriteImage(
                img_sitk,
                path.join(output_dir, f"{side}_{label}_heatmap.mhd"),
            )


def save_dataframe(centerline_dict: Dict[str, Dict[str, np.ndarray]], output_dir: str):

    output_df = pd.DataFrame(
        columns=[
            "side",
            "label",
            "z",
            "y",
            "x",
            "image_uncertainty",
            "heatmap_uncertainty",
        ]
    )
    for side in side_list:
        for label in ["internal", "external"]:
            markers = centerline_dict[side][label]
            row_df = pd.DataFrame(
                markers,
                columns=["z", "y", "x", "image_uncertainty", "heatmap_uncertainty"],
            )
            row_df["side"] = side
            row_df["label"] = label

            output_df = pd.concat([output_df, row_df])

    output_df.to_csv(path.join(output_dir, "centerline.tsv"), sep="\t", index=False)
