import torch
from tqdm import tqdm
from os import path, listdir
from typing import Dict, Any
from carotid.utils.transforms import ExtractLeftAndRightd, BuildEmptyHeatmapd, unravel_indices

from monai.transforms import Spacingd, Compose, InvertibleTransform, Orientationd
from monai.networks.nets import UNet
from monai.inferers import SlidingWindowInferer


class UNetPredictor:
    """
    Transform MRI, compute the heatmap, and process it back to the original space.
    """

    def __init__(self, parameters: Dict[str, Any]):

        # Remember all args
        self.model_dir = parameters["model_dir"]
        self.spacing = parameters["spacing"]
        self.device = parameters["device"]
        self.roi_size = parameters["roi_size"]

        # Create useful objects
        self.model_paths_list = [
            path.join(self.model_dir, filename)
            for filename in listdir(self.model_dir)
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

        self.transforms = self.get_transforms(spacing=self.spacing)
        self.inferer = SlidingWindowInferer(roi_size=self.roi_size, overlap=0.8)
        self.side_list = ["left", "right"]

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        # Remember original image to avoid a loss of resolution
        image_pt = sample["image"].clone()
        # Put the sample in the working space of the U-Net
        unet_sample = self.transforms(sample)

        for side in self.side_list:
            unet_shape = unet_sample[f"{side}_heatmap"].shape
            pred_tensor = torch.zeros(len(self.model_paths_list), *unet_shape)

            for model_idx, model_path in tqdm(
                enumerate(self.model_paths_list), desc="Predicting heatmaps", leave=False
            ):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()

                # Sides cannot be combined in the same batch as left and right images may have different dimensions
                with torch.no_grad():
                    prediction = (
                        self.inferer(
                            unet_sample[f"{side}_image"].unsqueeze(0).to(self.device),
                            self.model,
                        )
                        .squeeze(0)
                        .cpu()
                    )
                    pred_tensor[model_idx] = prediction

            unet_sample[f"{side}_heatmap"] += pred_tensor.reshape(len(self.model_paths_list) * 2, *unet_shape[1::])

        orig_sample = self.transforms.inverse(unet_sample)

        for side in self.side_list:
            batch_heatmap_pt = orig_sample[f"{side}_heatmap"].reshape(len(self.model_paths_list), 2, *image_pt.shape[1::])
            orig_sample[f"{side}_heatmap"] = {
                "mean": torch.mean(batch_heatmap_pt, dim=0),
                "std": torch.std(batch_heatmap_pt, dim=0),
            }
            flat_heatmap_pt = batch_heatmap_pt.reshape(len(self.model_paths_list), 2, -1, image_pt.shape[-1])
            flat_indices = torch.argmax(flat_heatmap_pt, dim=-2)
            indices = unravel_indices(flat_indices, image_pt.shape[1:3])
            orig_sample[f"{side}_heatmap"]["max_indices"] = indices
        orig_sample["image"] = image_pt

        return orig_sample

    def get_transforms(
        self, spacing: bool = False,
    ) -> InvertibleTransform:
        """
        Outputs a series of invertible transforms to go from the original space to the
        working space of the U-Net.
        """

        key_list = ["image", "left_heatmap", "right_heatmap"]
        transforms = [
            BuildEmptyHeatmapd(image_key="image", n_channels=len(self.model_paths_list) * 2),
            Orientationd(keys=key_list, axcodes="SPL"),
        ]
        if spacing:
            transforms.append(Spacingd(keys=key_list, pixdim=[0.5, 0.5, 0.5]))

        transforms.append(
            ExtractLeftAndRightd(split_keys=["heatmap"], keys=["image", "heatmap"])
        )

        return Compose(transforms)
