import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import Orientationd, RandAffined
from monai.data import MetaTensor
from copy import deepcopy

from carotid.utils import build_dataset
from carotid.utils.transforms import ExtractLeftAndRightd
from typing import Dict, Any, Tuple, Literal


def compute_centerline_df(
    raw_dir: str,
    centerline_dir: str,
) -> pd.DataFrame:
    centerline_dataset = build_dataset(raw_dir=raw_dir, centerline_dir=centerline_dir)
    centerline_df = pd.DataFrame()
    for sample in centerline_dataset:
        participant_id = sample["participant_id"]
        rows_df = pd.DataFrame(
            [[participant_id, side] for side in ["left", "right"]],
            columns=["participant_id", "side"],
        )
        centerline_df = pd.concat((centerline_df, rows_df))

    centerline_df.reset_index(inplace=True, drop=True)
    return centerline_df


class AnnotatedRawDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        centerline_dir: str,
        centerline_df: pd.DataFrame,
        params: Dict[str, Any],
        interleaved_augmentation: bool = True,
    ):
        """
        Builds a Dataset of annotated raw images.

        Args:
            raw_dir: path to the raw image.
            centerline_dir: path to the folder containing the annotations of the centerlines.
            centerline_df: DataFrame containing the list of all the centerlines which should be used.
            params: parameters of dataset.
            interleaved_augmentation: if True, different versions of interleaved axial slices will be sampled.
        """
        super().__init__()
        self.raw_dir = raw_dir
        self.centerline_dir = centerline_dir

        self.centerline_df = centerline_df
        self.params = params
        # training_space : "interleaved"
        self.interleaved_augmentation = interleaved_augmentation

        participant_list = centerline_df.participant_id.unique()
        self.cache_dataset = build_dataset(
            raw_dir=raw_dir,
            centerline_dir=centerline_dir,
            participant_list=participant_list,
        )

        ordered_participant_list = list()
        for sample in self.cache_dataset:
            ordered_participant_list.append(sample["participant_id"])

        self.participant_list = ordered_participant_list
        self.orientation_transform = Orientationd(keys=["image"], axcodes="SPL")
        self.rand_rotation = RandAffined(
            keys=["image"], rotate_range=0.1, translate_range=2, prob=1
        )
        self.split_transform = ExtractLeftAndRightd(keys=["image", "label"])

    def __len__(self):
        return len(self.centerline_df)

    def __getitem__(self, idx: int):
        participant_id = self.centerline_df.loc[idx, "participant_id"]
        side = self.centerline_df.loc[idx, "side"]
        participant_idx = self.participant_list.index(participant_id)
        sample = self.cache_dataset[participant_idx]
        assert sample["participant_id"] == participant_id

        raw_pt, heatmap_pt = self.compute_heatmap(sample, side)

        return {
            "participant_id": participant_id,
            "image": raw_pt,
            "label": heatmap_pt,
        }

    def compute_heatmap(
        self, sample: Dict[str, Any], side: Literal["side", "right"]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Crop the image to obtain the correct side.
        Compute the heatmap from centers.
        Crop both images to the annotated part along the z axis.
        Create interleaved version if needed.
        """
        # Work in SPL orientation as this is what was primarily chosen
        sample = self.orientation_transform(sample)

        # Generate heatmap in SPL space
        centerline_df = sample[f"{side}_centerline"]
        heatmap_pt = self.from_point_cloud_to_heatmap(
            centerline_df,
            heatmap_shape=sample["image"].shape,
            shape_parameter=self.params["shape_parameter"],
            intensity_parameter=self.params["intensity_parameter"],
        )
        sample["label"] = MetaTensor(heatmap_pt, meta=sample["image"].meta)

        # Crop unlabeled

        if self.interleaved_augmentation:
            odd_sample = self.rand_rotation(deepcopy(sample))
            even_sample = self.rand_rotation(deepcopy(sample))

            for key in ["image", "label"]:
                original_shape = odd_sample[key].shape
                combined_shape = (
                    original_shape[0],
                    original_shape[1] // 5,
                    original_shape[2],
                    original_shape[3],
                )
                combined_image = torch.zeros(combined_shape)
                combined_image[:, ::2] = even_sample[key][:, ::10]
                combined_image[:, 1::2] = odd_sample[key][:, 5::10]
                sample[key] = MetaTensor(combined_image, meta=sample[key])

        # Split sample

        return raw_pt, heatmap_pt

    @staticmethod
    def from_point_cloud_to_heatmap(
        centerline_df: pd.DataFrame,
        heatmap_shape: Tuple[int, int, int],
        shape_parameter: float = 32,
        intensity_parameter: float = 6,
    ) -> torch.Tensor:
        """
        Transforms continuous point cloud on a heatmap of the same size as the raw 3D image.

        Args:
            centerline_df: continuous annotations along z axis.
            heatmap_shape: spatial shape of the heatmap after extracting a side.
            shape_parameter: parameter modulating the locations where the heatmap is positive/negative.
                The higher its value, the larger the disks around the centers.
            intensity_parameter: parameter modulating the maximal value of the heatmap.

        Returns:
            heatmap used to train a U-Net learning to track the centerline.
        """
        heatmap_pt = torch.zeros((2, *heatmap_shape))
        x_coords, y_coords = torch.arange(heatmap_shape[-1]), torch.arange(
            heatmap_shape[-2]
        )
        grid = torch.stack(torch.meshgrid([y_coords, x_coords]))
        for idx in centerline_df.index.values:  # build heatmap slice-by-slice
            label = centerline_df.loc[idx, "label"]
            x = heatmap_shape[-1] - centerline_df.loc[idx, "x"]
            y = heatmap_shape[-2] - centerline_df.loc[idx, "y"]
            z = int(round(centerline_df.loc[idx, "z"], 0))

            center = torch.Tensor([y, x])
            dist_to_center = torch.linalg.norm(grid - center.reshape(2, 1, 1), axis=0)
            # Draw disk around circle with intensities having an exponential decat
            if label == "internal" or label == "common":
                heatmap_pt[0, z, :, :] = (
                    torch.exp(
                        intensity_parameter * (1 - dist_to_center / shape_parameter)
                    )
                    - 1
                ) * (dist_to_center < shape_parameter)
            if label == "external" or label == "common":
                heatmap_pt[1, z, :, :] = (
                    torch.exp(
                        intensity_parameter * (1 - dist_to_center / shape_parameter)
                    )
                    - 1
                ) * (dist_to_center < shape_parameter)

        return heatmap_pt
