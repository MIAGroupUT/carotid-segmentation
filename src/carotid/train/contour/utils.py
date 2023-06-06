import monai
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import scipy.spatial as scsp

from carotid.utils.data import build_dataset


class AnnotatedPolarDataset(monai.data.Dataset):

    def __init__(
        self,
        raw_dir: str,
        contour_dir: str,
        contour_df: pd.DataFrame,
        polar_params: Dict[str, Any],
        augmentation: bool = True,
    ):
        """
        Builds a Dataset of annotated polar images.

        Args:
            raw_dir: path to the raw image.
            contour_dir: path to the folder containing the annotations of the contours.
            contour_df: DataFrame containing the list of all the contours which should be used.
            polar_params: parameters of the polar transform.
            augmentation: if True, different centers will be sampled.
        """
        self.raw_dir = raw_dir
        self.contour_dir = contour_dir

        self.contour_df = contour_df
        self.polar_params = polar_params
        self.augmentation = augmentation
        self.polar_transform = PolarTransform(polar_params)

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
        annotation_df = self.polar_transform.transform_annotation(contour_df, center_pt)
        annotation_pt = torch.from_numpy(annotation_df.values)

        return polar_pt, annotation_pt

    def sample_center(self, contour_df: pd.DataFrame) -> torch.Tensor:
        lumen_df = contour_df[contour_df.object == "lumen"]
        lumen_np = np.array([lumen_df.x.mean(), lumen_df.y.mean(), lumen_df.z.unique().item()])
        max_rad = np.min(np.linalg.norm(lumen_np - lumen_df[["x", "y", "z"]].values, axis=1))
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


class PolarTransform:

    def __init__(self, polar_params: Dict[str, Any]):
        self.n_angles = polar_params["n_angles"]
        self.cartesian_ray = polar_params["cartesian_ray"]
        self.polar_ray = polar_params["polar_ray"]
        self.length = polar_params["length"]
        self.annotation_resolution = polar_params["annotation_resolution"]

        # construct coordinates of rays
        theta = torch.linspace(0, 2 * torch.pi, self.n_angles + 1)[: self.n_angles]
        radii = (
            torch.linspace(- float(self.cartesian_ray) / 2, float(self.cartesian_ray) / 2, self.polar_ray)
        )
        zz = torch.arange(
            -self.length // 2 + self.length % 2, self.length // 2 + 1, dtype=torch.float
        )
        R, Z, T = torch.meshgrid(radii, zz, theta, indexing="xy")
        xx = (R * torch.cos(T)).flatten()
        yy = (R * torch.sin(T)).flatten()
        self.coords3d = torch.stack([xx, yy, Z.flatten()], dim=1)  # n_angles * length * polar_ray, 3

        radii = torch.linspace(
            - float(self.cartesian_ray) / 2,
            float(self.cartesian_ray) / 2,
            self.annotation_resolution * self.polar_ray
        )
        R, T = torch.meshgrid(radii, theta, indexing="xy")
        xx = (R * torch.cos(T))
        yy = (R * torch.sin(T))
        self.coords2d = torch.stack([xx, yy], dim=-1)  # polar_ray, n_angles, 2

    def transform_image(self, image_pt: torch.Tensor, center_pt: torch.Tensor) -> torch.Tensor:
        coords = self.coords3d + center_pt
        polar_pt = self.fast_trilinear_interpolation(
            image_pt, coords[:, 0], coords[:, 1], coords[:, 2]
        ).reshape(self.length, self.polar_ray, self.n_angles)

        polar_pt = torch.cat(
            [
                polar_pt[:, :, self.n_angles // 2 + 1:],
                polar_pt,
                polar_pt[:, :, : self.n_angles // 2],
            ],
            dim=-1,
        ).transpose(1, 2)

        return polar_pt

    def transform_annotation(self, contour_df: pd.DataFrame, center_pt: torch.Tensor) -> pd.DataFrame:

        theta = torch.linspace(0, 2 * torch.pi, self.n_angles + 1)[: self.n_angles]
        lumen_np = contour_df[contour_df.object == "lumen"][["x", "y"]].values
        wall_np = contour_df[contour_df.object == "wall"][["x", "y"]].values

        output_df = pd.DataFrame(index=theta.numpy(), columns=["lumen_radius", "wall_width"], dtype=np.float32)
        for angle_idx, angle_val in enumerate(theta):
            coords = self.coords2d[angle_idx] + center_pt[:2]
            distmat_lumen = scsp.distance.cdist(
                coords[: (self.annotation_resolution * self.polar_ray // 2)], lumen_np
            )
            ij_min_lumen = np.unravel_index(distmat_lumen.argmin(), distmat_lumen.shape)

            distmat_wall = scsp.distance.cdist(
                coords[: (self.annotation_resolution * self.polar_ray // 2)], wall_np
            )
            ij_min_wall = np.unravel_index(distmat_wall.argmin(), distmat_wall.shape)

            output_df.loc[angle_val.item(), "lumen_radius"] = self.polar_ray // 2 - ij_min_lumen[0] / self.annotation_resolution
            output_df.loc[angle_val.item(), "wall_width"] = (ij_min_lumen[0] - ij_min_wall[0]) / self.annotation_resolution

        return output_df

    @staticmethod
    def fast_trilinear_interpolation(
            array_pt: torch.Tensor,
            x_indices: torch.Tensor,
            y_indices: torch.Tensor,
            z_indices: torch.Tensor,
        ) -> torch.Tensor:
            # interpolate the image for given input coordinates.

            x0 = x_indices.long()
            y0 = y_indices.long()
            z0 = z_indices.long()
            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1

            x0 = torch.clip(x0, 0, array_pt.shape[0] - 1)
            y0 = torch.clip(y0, 0, array_pt.shape[1] - 1)
            z0 = torch.clip(z0, 0, array_pt.shape[2] - 1)
            x1 = torch.clip(x1, 0, array_pt.shape[0] - 1)
            y1 = torch.clip(y1, 0, array_pt.shape[1] - 1)
            z1 = torch.clip(z1, 0, array_pt.shape[2] - 1)

            x = x_indices - x0
            y = y_indices - y0
            z = z_indices - z0

            output_pt = (
                array_pt[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
                + array_pt[x0, y0, z1] * x * (1 - y) * (1 - z)
                + array_pt[x0, y1, z0] * (1 - x) * y * (1 - z)
                + array_pt[x0, y1, z1] * x * y * (1 - z)
                + array_pt[x1, y0, z0] * (1 - x) * (1 - y) * z
                + array_pt[x1, y1, z0] * (1 - x) * y * z
                + array_pt[x1, y0, z1] * x * (1 - y) * z
                + array_pt[x1, y1, z1] * x * y * z
            )
            return output_pt
