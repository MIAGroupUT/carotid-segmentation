from typing import Dict, Any
import torch


# TODO solve problem of too low / high centerlines for 3D length


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

        # construct coordinates of rays
        theta = torch.linspace(0, 2 * torch.pi, self.n_angles + 1)[: self.n_angles]
        radii = (
            torch.linspace(0, float(self.cartesian_ray), self.polar_ray)
            - float(self.cartesian_ray) / 2
        )
        zz = torch.arange(
            -self.length // 2 + self.length % 2, self.length // 2 + 1, dtype=torch.float
        )
        R, Z, T = torch.meshgrid(radii, zz, theta, indexing="xy")
        xx = (R * torch.cos(T)).flatten()
        yy = (R * torch.sin(T)).flatten()
        self.coords = torch.stack([Z.flatten(), yy, xx], dim=1)

    def __call__(self, sample: Dict[str, Any]):
        image_pt = sample["image"][0]
        for side in ["left", "right"]:
            centerline_df = sample[f"{side}_centerline"]
            sample[f"{side}_polar"] = list()
            for idx in centerline_df.index.values:
                label_name = centerline_df.loc[idx, "label"]
                center_pt = torch.from_numpy(
                    centerline_df.loc[idx, ["z", "y", "x"]].values.astype(float)
                )
                polar_pt = self._transform(image_pt, center_pt)
                sample[f"{side}_polar"].append(
                    {
                        "label": label_name,
                        "slice_idx": int(center_pt[0]),
                        "polar_pt": polar_pt.unsqueeze(0).float(),
                        "center_pt": center_pt,
                    }
                )
            sample[f"{side}_polar_meta_dict"] = {
                "n_angles": self.n_angles,
                "polar_ray": self.polar_ray,
                "cartesian_ray": self.cartesian_ray,
                "length": self.length,
            }

        return sample

    def _transform(self, image_pt: torch.Tensor, center_pt: torch.Tensor):
        coords = self.coords + center_pt
        polar_transform = self.fast_trilinear_interpolation(
            image_pt, coords[:, 2], coords[:, 1], coords[:, 0]
        ).reshape(self.length, self.polar_ray, self.n_angles)
        return torch.cat(
            [
                polar_transform[:, :, self.n_angles // 2 + 1 :],
                polar_transform,
                polar_transform[:, :, : self.n_angles // 2],
            ],
            dim=-1,
        ).transpose(1, 2)

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

        x0 = torch.clip(x0, 0, array_pt.shape[2] - 1)
        y0 = torch.clip(y0, 0, array_pt.shape[1] - 1)
        z0 = torch.clip(z0, 0, array_pt.shape[0] - 1)
        x1 = torch.clip(x1, 0, array_pt.shape[2] - 1)
        y1 = torch.clip(y1, 0, array_pt.shape[1] - 1)
        z1 = torch.clip(z1, 0, array_pt.shape[0] - 1)

        x = x_indices - x0
        y = y_indices - y0
        z = z_indices - z0

        output_pt = (
            array_pt[z0, y0, x0] * (1 - x) * (1 - y) * (1 - z)
            + array_pt[z0, y0, x1] * x * (1 - y) * (1 - z)
            + array_pt[z0, y1, x0] * (1 - x) * y * (1 - z)
            + array_pt[z0, y1, x1] * x * y * (1 - z)
            + array_pt[z1, y0, x0] * (1 - x) * (1 - y) * z
            + array_pt[z1, y1, x0] * (1 - x) * y * z
            + array_pt[z1, y0, x1] * x * (1 - y) * z
            + array_pt[z1, y1, x1] * x * y * z
        )
        return output_pt
