import torch
from tqdm import tqdm
from os import path, listdir
from typing import Tuple, Dict, Any
from carotid.utils.transforms import ExtractLeftAndRightd, BuildEmptyHeatmapd

from monai.transforms import Flipd, Spacingd, Compose, InvertibleTransform, ToTensord
from monai.networks.nets import UNet
from monai.inferers import SlidingWindowInferer


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
        self.side_list = ["left", "right"]

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        # Remember original image to avoid a loss of resolution
        image_np = sample["img"].copy()
        # Put the sample in the working space of the U-Net
        unet_sample = self.transforms(sample)
        pred_dict = {
            side: torch.zeros_like(unet_sample[f"{side}_heatmap"])
            for side in self.side_list
        }

        # Compute sum of heatmaps for both sides
        for index, model_path in tqdm(
            enumerate(self.model_paths_list), desc="Predicting heatmaps", leave=False
        ):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

            # TODO: combine both sides on the same batch
            for side in self.side_list:
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
        for side in self.side_list:
            unet_sample[f"{side}_heatmap"] = pred_dict[side] / len(
                self.model_paths_list
            )

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

        key_list = ["img", "left_heatmap", "right_heatmap"]
        transforms = [BuildEmptyHeatmapd(image_key="img")]
        if flip_z:
            transforms.append(Flipd(keys=key_list, spatial_axis=0))
        if spacing:
            transforms.append(Spacingd(keys=key_list, pixdim=[0.5, 0.5, 0.5]))

        transforms.append(
            ExtractLeftAndRightd(split_keys=["heatmap"], keys=["img", "heatmap"])
        )
        transforms.append(
            ToTensord(keys=["left_img", "right_img", "left_heatmap", "right_heatmap"])
        )
        # TODO: add transpose

        return Compose(transforms)
