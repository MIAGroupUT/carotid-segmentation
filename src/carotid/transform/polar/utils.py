from typing import Dict, Any
import pandas as pd
import numpy as np
import scipy.spatial as scsp
import torch


class PolarTransform:
    def __init__(self, parameters: Dict[str, Any]):
        """
        Args:
            parameters: dictionary with following entries
                n_angles: number of angles to discretize the cylinder in
                polar_ray: number of intensities along ray
                cartesian_ray: diameter of cylinder in voxel coordinates
                length: length of cylinder in terms of z coordinates
        """
        self.n_angles = parameters["n_angles"]
        self.polar_ray = parameters["polar_ray"]
        self.cartesian_ray = parameters["cartesian_ray"]
        self.length = parameters["length"]
        self.multiple_centers = parameters["multiple_centers"]
        self.annotation_resolution = parameters["annotation_resolution"]
        self.side_list = ["left", "right"]
        self.dimension = 3 if "dimension" not in parameters else parameters["dimension"]

        # construct coordinates of rays
        theta = torch.linspace(0, 2 * torch.pi, self.n_angles + 1)[: self.n_angles]
        radii = torch.linspace(
            -float(self.cartesian_ray) / 2,
            float(self.cartesian_ray) / 2,
            self.polar_ray,
        )
        zz = torch.arange(
            -self.length // 2 + self.length % 2, self.length // 2 + 1, dtype=torch.float
        )

        # Build 3D coordinates
        R, Z, T = torch.meshgrid(radii, zz, theta, indexing="xy")
        xx = (R * torch.cos(T)).flatten()
        yy = (R * torch.sin(T)).flatten()
        self.coords3d = torch.stack(
            [xx, yy, Z.flatten()], dim=1
        )  # n_angles * length * polar_ray, 3

        radii = torch.linspace(
            -float(self.cartesian_ray) / 2,
            float(self.cartesian_ray) / 2,
            self.annotation_resolution * self.polar_ray,
        )
        R, T = torch.meshgrid(radii, theta, indexing="xy")
        xx = R * torch.cos(T)
        yy = R * torch.sin(T)
        self.coords_annotation = torch.stack([xx, yy], dim=-1)  # polar_ray, n_angles, 2

    def __call__(self, sample: Dict[str, Any]):
        image_pt = sample["image"][0]
        for side in self.side_list:
            centerline_df = sample[f"{side}_centerline"]
            sample[f"{side}_polar"] = list()
            for idx in centerline_df.index.values:
                label_name = centerline_df.loc[idx, "label"]
                center_pt = torch.from_numpy(
                    centerline_df.loc[idx, ["x", "y", "z"]].values.astype(float)
                )
                if self.multiple_centers:
                    batch_center_pt = self._sample_centers(center_pt)
                else:
                    batch_center_pt = center_pt.unsqueeze(0)
                batch_polar_pt = self._transform(image_pt, batch_center_pt)
                sample[f"{side}_polar"].append(
                    {
                        "label": label_name,
                        "slice_idx": int(center_pt[2]),
                        "polar_pt": batch_polar_pt.float(),
                        "center_pt": batch_center_pt,
                    }
                )
            sample[f"{side}_polar_meta_dict"] = {
                "n_angles": self.n_angles,
                "polar_ray": self.polar_ray,
                "cartesian_ray": self.cartesian_ray,
                "length": self.length,
                "orig_shape": image_pt.shape,
                "affine": image_pt.affine.tolist(),
            }

        return sample

    @staticmethod
    def _sample_centers(center_pt: torch.Tensor) -> torch.Tensor:
        """
        Samples a list of centers around the original one.

        Args:
            center_pt: Position of the original center.

        Returns:
            A tensor corresponding to a batch of centers of size.
        """
        batch_center_pt = center_pt.repeat(9, 1)
        batch_idx = 0

        for x_offset in [0, -1, 1]:
            for y_offset in [0, -1, 1]:
                batch_center_pt[batch_idx, 0] += x_offset
                batch_center_pt[batch_idx, 1] += y_offset
                batch_idx += 1

        return batch_center_pt

    def _transform(
        self, image_pt: torch.Tensor, batch_center_pt: torch.Tensor
    ) -> torch.Tensor:

        batch_polar_pt = torch.zeros(
            (
                len(batch_center_pt),
                1,
                self.length,
                2 * self.n_angles - 1,
                self.polar_ray,
            )
        )
        for center_idx, center_pt in enumerate(batch_center_pt):
            polar_pt = self.transform_image(image_pt, center_pt)
            batch_polar_pt[center_idx, 0] = polar_pt

        return batch_polar_pt

    # TODO: implement more efficient cylindrical interpolation
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

    def transform_image(
        self, image_pt: torch.Tensor, center_pt: torch.Tensor
    ) -> torch.Tensor:
        coords = self.coords3d + center_pt
        polar_pt = self.fast_trilinear_interpolation(
            image_pt, coords[:, 0], coords[:, 1], coords[:, 2]
        ).reshape(self.length, self.polar_ray, self.n_angles)

        polar_pt = torch.cat(
            [
                polar_pt[:, :, self.n_angles // 2 + 1 :],
                polar_pt,
                polar_pt[:, :, : self.n_angles // 2],
            ],
            dim=-1,
        ).transpose(1, 2)

        return polar_pt

    def transform_annotation(
        self, contour_df: pd.DataFrame, center_pt: torch.Tensor
    ) -> pd.DataFrame:

        theta = torch.linspace(0, 2 * torch.pi, self.n_angles + 1)[: self.n_angles]
        lumen_np = contour_df[contour_df.object == "lumen"][["x", "y"]].values
        wall_np = contour_df[contour_df.object == "wall"][["x", "y"]].values

        output_df = pd.DataFrame(
            index=theta.numpy(),
            columns=["lumen_radius", "wall_width"],
            dtype=np.float32,
        )
        for angle_idx, angle_val in enumerate(theta):
            coords = self.coords_annotation[angle_idx] + center_pt[:2]
            distmat_lumen = scsp.distance.cdist(
                coords[: (self.annotation_resolution * self.polar_ray // 2)], lumen_np
            )
            ij_min_lumen = np.unravel_index(distmat_lumen.argmin(), distmat_lumen.shape)

            distmat_wall = scsp.distance.cdist(
                coords[: (self.annotation_resolution * self.polar_ray // 2)], wall_np
            )
            ij_min_wall = np.unravel_index(distmat_wall.argmin(), distmat_wall.shape)

            output_df.loc[angle_val.item(), "lumen_radius"] = (
                self.polar_ray // 2 - ij_min_lumen[0] / self.annotation_resolution
            )
            output_df.loc[angle_val.item(), "wall_width"] = (
                ij_min_lumen[0] - ij_min_wall[0]
            ) / self.annotation_resolution

        return output_df
