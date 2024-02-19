from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from carotid.transform.polar.utils import PolarTransform
from carotid.utils.data import build_dataset


def compute_contour_df(
    raw_dir: str,
    contour_dir: str,
) -> pd.DataFrame:
    contour_dataset = build_dataset(raw_dir=raw_dir, contour_dir=contour_dir)
    contour_df = pd.DataFrame()
    for sample in contour_dataset:
        participant_id = sample["participant_id"]
        for side in ["left", "right"]:
            df = sample[f"{side}_contour"]
            df = df[["label", "z"]].drop_duplicates()
            df["participant_id"] = participant_id
            df["side"] = side
            contour_df = pd.concat((contour_df, df))

    contour_df.reset_index(inplace=True, drop=True)
    return contour_df


class AnnotatedPolarDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        contour_dir: str,
        contour_df: pd.DataFrame,
        polar_params: Dict[str, Any],
        augmentation: bool = True,
        dimension: int = 3,
    ):
        """
        Builds a Dataset of annotated polar images.

        Args:
            raw_dir: path to the raw image.
            contour_dir: path to the folder containing the annotations of the contours.
            contour_df: DataFrame containing the list of all the contours which should be used.
            polar_params: parameters of the polar transform.
            augmentation: if True, different centers will be sampled.
            dimension: dimension of the output polar maps
        """
        super().__init__()
        self.raw_dir = raw_dir
        self.contour_dir = contour_dir

        self.contour_df = contour_df
        self.polar_params = polar_params
        self.augmentation = augmentation
        self.polar_transform = PolarTransform(polar_params)
        self.dimension = dimension

        participant_list = contour_df.participant_id.unique()
        self.cache_dataset = build_dataset(
            raw_dir=raw_dir,
            contour_dir=contour_dir,
            participant_list=participant_list,
        )

        ordered_participant_list = list()
        for sample in self.cache_dataset:
            ordered_participant_list.append(sample["participant_id"])

        self.participant_list = ordered_participant_list

    def __len__(self):
        return len(self.contour_df)

    def __getitem__(self, idx: int):
        participant_id = self.contour_df.loc[idx, "participant_id"]
        side = self.contour_df.loc[idx, "side"]
        label = self.contour_df.loc[idx, "label"]
        slice_idx = self.contour_df.loc[idx, "z"]

        participant_idx = self.participant_list.index(participant_id)
        sample = self.cache_dataset[participant_idx]
        assert sample["participant_id"] == participant_id

        side_df = sample[f"{side}_contour"]
        contour_df = side_df[(side_df.label == label) & (side_df.z == slice_idx)]
        raw_image_pt = sample["image"][0]

        polar_image_pt, annotations_pt = self.compute_polar(
            raw_image_pt=raw_image_pt,
            contour_df=contour_df,
        )

        return {
            "participant_id": participant_id,
            "side": side,
            "label": label,
            "z": slice_idx,
            "image": polar_image_pt.unsqueeze(0).float(),
            "labels": annotations_pt,
        }

    def compute_polar(
        self, raw_image_pt: torch.Tensor, contour_df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        center_pt = self.sample_center(contour_df)
        polar_pt = self.polar_transform.transform_image(raw_image_pt, center_pt)
        if self.dimension == 2:
            polar_pt = polar_pt[self.polar_params["length"] // 2]

        annotation_df = self.polar_transform.transform_annotation(contour_df, center_pt)
        annotation_pt = torch.from_numpy(annotation_df.values)

        return polar_pt, annotation_pt

    def sample_center(self, contour_df: pd.DataFrame) -> torch.Tensor:
        lumen_df = contour_df[contour_df.object == "lumen"]
        lumen_np = np.array(
            [lumen_df.x.mean(), lumen_df.y.mean(), lumen_df.z.unique().item()]
        )
        max_rad = np.min(
            np.linalg.norm(lumen_np - lumen_df[["x", "y", "z"]].values, axis=1)
        )
        # TODO change the range of sampling for the center (half of the minimum radius may not be appropriate)

        if self.augmentation:
            center_pt = torch.zeros(3)
            r = np.random.uniform(0, max_rad / 2)
            theta = np.random.uniform(0, 2 * np.pi)
            center_pt[0] = r * np.cos(theta) + lumen_np[0]
            center_pt[1] = r * np.sin(theta) + lumen_np[1]
            center_pt[2] = lumen_np[2]
        else:
            center_pt = torch.from_numpy(lumen_np)

        return center_pt
