import numpy as np
from typing import Dict, Any


class PolarTransform:
    def __init__(self, n_angles: int, polar_ray: int, cartesian_ray: int, length: int):
        """
        Args:
            n_angles: number of angles to discretize the cylinder in
            polar_ray: number of intensities along ray
            cartesian_ray: diameter of cylinder in voxel coordinates
            length: length of cylinder in terms of z coordinates
        """
        self.n_angles = n_angles
        self.polar_ray = polar_ray
        self.cartesian_ray = cartesian_ray
        self.length = length

        # construct coordinates of rays
        theta = np.linspace(0, 2 * np.pi, self.n_angles + 1)[: self.n_angles]
        radii = (
            np.linspace(0, float(self.cartesian_ray), self.polar_ray)
            - float(self.cartesian_ray) / 2
        )
        zz = np.arange(np.ceil(-self.length / 2), self.length // 2 + 1)
        R, Z, T = np.meshgrid(radii, zz, theta, indexing="xy")
        xx = (R * np.cos(T)).flatten()
        yy = (R * np.sin(T)).flatten()
        self.coords = np.stack([Z.flatten(), yy, xx], axis=1)

    def __call__(self, sample: Dict[str, Any]):
        image_np = sample["img"][0]
        for side in ["left", "right"]:
            centerline_df = sample[f"{side}_centerline"]
            sample[f"{side}_polar"] = list()
            for idx in centerline_df.index.values:
                label_name = centerline_df.loc[idx, "label"]
                center = centerline_df.loc[idx, ["z", "y", "x"]].values
                polar_np = self._transform(image_np, center)
                sample[f"{side}_polar"].append(
                    {
                        "label": label_name,
                        "slice_idx": int(center[0]),
                        "polar_img": polar_np,
                    }
                )

        return sample

    def _transform(self, image_np: np.ndarray, center_np: np.ndarray):
        # does a polar transform on the image, given the center. Everything is in image coordinates!
        coords = self.coords + center_np
        polar_transform = fast_trilinear_interpolation(
            image_np, coords[:, 2], coords[:, 1], coords[:, 0]
        ).reshape(self.length, self.polar_ray, self.n_angles)
        return np.concatenate(
            [
                polar_transform[:, :, self.n_angles // 2 + 1 :],
                polar_transform,
                polar_transform[:, :, : self.n_angles // 2],
            ],
            axis=-1,
        ).transpose((0, 2, 1))


# TODO: implement more efficient cylindrical interpolation
def fast_trilinear_interpolation(
    array_np: np.ndarray,
    x_indices: np.ndarray,
    y_indices: np.ndarray,
    z_indices: np.ndarray,
):
    # interpolate the image for given input coordinates.

    x0 = x_indices.astype(int)
    y0 = y_indices.astype(int)
    z0 = z_indices.astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = np.clip(x0, 0, array_np.shape[2] - 1)
    y0 = np.clip(y0, 0, array_np.shape[1] - 1)
    z0 = np.clip(z0, 0, array_np.shape[0] - 1)
    x1 = np.clip(x1, 0, array_np.shape[2] - 1)
    y1 = np.clip(y1, 0, array_np.shape[1] - 1)
    z1 = np.clip(z1, 0, array_np.shape[0] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        array_np[z0, y0, x0] * (1 - x) * (1 - y) * (1 - z)
        + array_np[z0, y0, x1] * x * (1 - y) * (1 - z)
        + array_np[z0, y1, x0] * (1 - x) * y * (1 - z)
        + array_np[z0, y1, x1] * x * y * (1 - z)
        + array_np[z1, y0, x0] * (1 - x) * (1 - y) * z
        + array_np[z1, y1, x0] * (1 - x) * y * z
        + array_np[z1, y0, x1] * x * (1 - y) * z
        + array_np[z1, y1, x1] * x * y * z
    )
    return output
