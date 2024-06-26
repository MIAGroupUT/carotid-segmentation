import warnings
from typing import Dict, Any, Tuple, List, Union
from carotid.utils.transforms import polar2cart, cart2polar
import pandas as pd
from torch import nn
import torch
import numpy as np
from os import path, listdir
from logging import getLogger


logger = getLogger("carotid")


class ContourTransform:
    def __init__(self, parameters: Dict[str, Any]):
        self.model_dir = parameters["model_dir"]
        self.device = parameters["device"]
        self.dropout = parameters["dropout"]
        self.n_repeats = parameters["n_repeats"]
        self.side_list = ["left", "right"]
        self.delta_theta = 2 * np.pi * parameters["delta_theta"]
        self.single_center = parameters["single_center"]
        self.split_list = parameters["split_list"]

        if self.single_center:
            self.interpolation_method = parameters["interpolation_method"]
        else:
            self.interpolation_method = "polynomial"

        # Find model paths
        if path.exists(path.join(self.model_dir, "parameters.json")):
            if self.split_list is None or len(self.split_list) == 0:
                self.split_list = listdir(self.model_dir)

            self.model_paths_list = [
                path.join(self.model_dir, split_dir, "model.pt")
                for split_dir in self.split_list
                if split_dir.startswith("split-")
                and path.exists(path.join(self.model_dir, split_dir, "model.pt"))
            ]
            self.version = 2
        else:
            self.model_paths_list = [
                path.join(self.model_dir, filename)
                for filename in listdir(self.model_dir)
                if filename.endswith(".pt")
            ]
            self.version = 1

        # Choose model architecture
        if self.version == 1:
            trained_weights = torch.load(
                self.model_paths_list[0], map_location=self.device
            )
        else:
            trained_weights = torch.load(
                self.model_paths_list[0], map_location=self.device
            )["model"]

        # Dropout model has more layers
        model_dropout = "model.27.bias" in trained_weights.keys()
        model_dimension = len(trained_weights["model.0.weight"].shape) - 2
        self.dimension = model_dimension

        if self.dropout and not model_dropout:
            warnings.warn(
                "Dropout resampling was activated but the models do not contain "
                "dropout layers. The dropout resampling is deactivated."
            )
            self.dropout = False

        self.model = ContourNet(dropout=model_dropout, dimension=model_dimension).to(
            self.device
        )

        if not self.dropout:
            self.n_repeats = 1

        self.model.set_mode("dropout" if self.dropout else "eval")

    def __call__(self, sample):
        for side in self.side_list:
            contour_df = pd.DataFrame(
                columns=[
                    "label",
                    "object",
                    "z",
                    "y",
                    "x",
                    "deviation",
                    "norm_deviation",
                ]
            )
            polar_list = sample[f"{side}_polar"]
            orig_shape = sample[f"{side}_polar_meta_dict"]["orig_shape"]
            affine = sample[f"{side}_polar_meta_dict"]["affine"]
            for polar_dict in polar_list:
                label = polar_dict["label"]
                if self.single_center:
                    batch_polar_pt = polar_dict["polar_pt"][:1:]
                    batch_center_pt = polar_dict["center_pt"][:1:]
                else:
                    batch_polar_pt = polar_dict["polar_pt"]
                    batch_center_pt = polar_dict["center_pt"]

                # Remove z dimension if 2D
                if self.dimension == 2:
                    length = batch_polar_pt.shape[2]
                    batch_polar_pt = batch_polar_pt[:, :, length // 2]

                lumen_cont, wall_cont = self._transform(
                    batch_polar_pt,
                    batch_center_pt,
                    sample[f"{side}_polar_meta_dict"],
                )

                # Build DataFrames
                lumen_slice_df = pd.DataFrame(
                    data=lumen_cont.numpy(),
                    columns=["z", "y", "x", "deviation", "norm_deviation"],
                )
                wall_slice_df = pd.DataFrame(
                    data=wall_cont.numpy(),
                    columns=["z", "y", "x", "deviation", "norm_deviation"],
                )
                lumen_slice_df["object"] = "lumen"
                wall_slice_df["object"] = "wall"
                lumen_slice_df["label"] = label
                wall_slice_df["label"] = label

                contour_df = pd.concat((contour_df, lumen_slice_df, wall_slice_df))
            contour_df.reset_index(drop=True, inplace=True)
            sample[f"{side}_contour"] = contour_df
            sample[f"{side}_contour_meta_dict"] = {
                "orig_shape": orig_shape,
                "affine": affine,
            }

        return sample

    def _transform(
        self,
        batch_polar_pt: torch.Tensor,
        batch_center_pt: torch.Tensor,
        polar_meta_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_angles = polar_meta_dict["n_angles"]
        polar_ray = polar_meta_dict["polar_ray"]
        cartesian_ray = polar_meta_dict["cartesian_ray"]

        if len(self.model_paths_list) > 1:
            batch_prediction_pt = self.predict_multiple_models(batch_polar_pt)
        else:
            batch_prediction_pt = self.sample_single_models(batch_polar_pt)

        if self.interpolation_method == "polynomial":
            lumen_cloud, wall_cloud = self.pred2cartesian_batch(
                batch_prediction_pt,
                batch_center_pt=batch_center_pt,
                n_angles=n_angles,
                polar_ray=polar_ray,
                cartesian_ray=cartesian_ray,
                version=self.version,
            )

            lumen_cont = self.interpolate_contour(
                lumen_cloud, n_angles, self.delta_theta
            )
            wall_cont = self.interpolate_contour(wall_cloud, n_angles, self.delta_theta)
        else:
            lumen_cont, wall_cont = self.mean_distance_per_angle(
                batch_prediction_pt,
                batch_center_pt=batch_center_pt,
                n_angles=n_angles,
                polar_ray=polar_ray,
                cartesian_ray=cartesian_ray,
            )

        return lumen_cont, wall_cont

    def mean_distance_per_angle(
        self,
        batch_prediction_pt: torch.Tensor,
        batch_center_pt: torch.Tensor,
        n_angles: int,
        polar_ray: int,
        cartesian_ray: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reduces several contours with the same center to one contour with an uncertainty estimation
        corresponding to the standard deviation of the distances.
        """
        batch_size, n_pred, n_coords, n_angles_pred = batch_prediction_pt.shape
        assert n_coords == 2
        assert n_angles_pred == n_angles
        assert batch_size == 1

        mean_pred_pt = torch.mean(batch_prediction_pt, dim=1)
        mean_lumen_pt, mean_wall_pt = self.pred2cartesian_single(
            mean_pred_pt,
            batch_center_pt.squeeze(),
            n_angles=n_angles,
            polar_ray=polar_ray,
            cartesian_ray=cartesian_ray,
            version=self.version,
        )
        std_pred_pt = torch.std(batch_prediction_pt, dim=1).unsqueeze(-1)
        lumen_pt = torch.hstack(
            (
                mean_lumen_pt[0],
                std_pred_pt[0, 0, :],
                std_pred_pt[0, 0, :] / mean_pred_pt[0, 0, :].unsqueeze(-1),
            )
        )
        wall_pt = torch.hstack(
            (
                mean_wall_pt[0],
                std_pred_pt[0, 1, :],
                std_pred_pt[0, 1, :] / mean_pred_pt[0, 1, :].unsqueeze(-1),
            )
        )

        return lumen_pt, wall_pt

    @staticmethod
    def pred2cartesian_single(
        prediction_pt,
        center_pt,
        n_angles: int,
        polar_ray: int,
        cartesian_ray: int,
        version: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms a set of n_pred predictions corresponding to one single center in points in cartesian coordinates.

        Args:
            prediction_pt: tensor of size (n_pred, 2, n_angles).
            center_pt: single center, tensor of size (3,).
            n_angles: number of angles to check the validity of the input.
            polar_ray: length of the polar ray.
            cartesian_ray: length of the cartesian ray.
            version: Version number of the model. Version 2 considers that the first prediction is the actual
                radius of the lumen, whereas the first is the distance between the lumen and the border of the image.

        Returns:
            lumen_cont: contour of the lumen in cartesian coordinates (n_pred, n_angles, 3)
            wall_cont: contour of the wall in cartesian coordinates (n_pred, n_angles, 3)
        """
        n_pred, n_coords, n_angles_pred = prediction_pt.shape
        assert n_coords == 2
        assert n_angles_pred == n_angles

        lumen_cont = torch.zeros((n_pred, n_angles, 3))
        wall_cont = torch.zeros_like(lumen_cont)

        # First coordinate of prediction is the distance between the border and the lumen
        if version == 1:
            lumen_dists = (
                (polar_ray / 2 - prediction_pt[:, 0, :]) * cartesian_ray / polar_ray
            )
        else:
            lumen_dists = prediction_pt[:, 0, :] * cartesian_ray / polar_ray
        # Second coordinate of prediction is the wall width
        wall_dists = lumen_dists + prediction_pt[:, 1, :] * cartesian_ray / polar_ray

        angle_vals = torch.linspace(-np.pi, np.pi, (n_angles + 1))[:n_angles].reshape(
            1, -1
        )
        lumen_cont[:, :, 0] = center_pt[2]
        lumen_cont[:, :, 1] = lumen_dists * torch.sin(angle_vals) + center_pt[1]
        lumen_cont[:, :, 2] = lumen_dists * torch.cos(angle_vals) + center_pt[0]
        wall_cont[:, :, 0] = center_pt[2]
        wall_cont[:, :, 1] = wall_dists * torch.sin(angle_vals) + center_pt[1]
        wall_cont[:, :, 2] = wall_dists * torch.cos(angle_vals) + center_pt[0]

        return lumen_cont, wall_cont

    @staticmethod
    def pred2cartesian_batch(
        batch_prediction_pt: torch.Tensor,
        batch_center_pt: torch.Tensor,
        n_angles: int,
        polar_ray: int,
        cartesian_ray: int,
        version: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps N batch of polar predictions of the CNN to the cartesian space.

        Each polar prediction is of size (2, n_angles):
            - First column corresponds to be the distance between the left border of the image and the lumen.
            - Second column is the wall width.
        Corresponding angles correspond to an equal split of the trigonometrical circle in n_angles.
        """
        batch_size, n_pred, n_coords, n_angles_pred = batch_prediction_pt.shape
        assert n_coords == 2
        assert n_angles_pred == n_angles

        lumen_cont = torch.zeros((batch_size, n_pred, n_angles, 3))
        wall_cont = torch.zeros_like(lumen_cont)
        for batch_idx, (prediction_pt, center_pt) in enumerate(
            zip(batch_prediction_pt, batch_center_pt)
        ):
            (
                lumen_cont[batch_idx],
                wall_cont[batch_idx],
            ) = ContourTransform.pred2cartesian_single(
                prediction_pt,
                center_pt,
                n_angles=n_angles,
                polar_ray=polar_ray,
                cartesian_ray=cartesian_ray,
                version=version,
            )

        lumen_cont = lumen_cont.reshape((batch_size * n_pred * n_angles, 3))
        wall_cont = wall_cont.reshape((batch_size * n_pred * n_angles, 3))

        return lumen_cont, wall_cont

    def predict_multiple_models(self, batch_polar_pt: torch.Tensor) -> torch.Tensor:
        """
        Ensemble a batch of polar map corresponding to a batch of centers using
        different models, eventually repeated for dropout sampling.

        Args:
            batch_polar_pt: batch of polar maps.

        Returns:
            batch of prediction of size (batch_size, n_models * n_repeats, 2, n_angles)
        """
        batch_size, dn_angles = batch_polar_pt.shape[0], batch_polar_pt.shape[-2]
        batch_prediction_pt = torch.zeros(
            (
                len(self.model_paths_list),
                batch_size,
                self.n_repeats,
                2,
                dn_angles // 2 + 1,
            )
        )

        for model_index, model_path in enumerate(self.model_paths_list):
            logger.debug(f"Applying model at path {model_path}")
            if self.version == 1:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
            else:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)["model"]
                )
            self.model.set_mode("dropout" if self.dropout else "eval")

            with torch.no_grad():
                batch_prediction_pt[model_index] = self.sample_single_models(
                    batch_polar_pt
                )

        batch_prediction_pt = batch_prediction_pt.transpose(0, 1).reshape(
            batch_size, -1, 2, dn_angles // 2 + 1
        )

        return batch_prediction_pt

    def sample_single_models(self, batch_polar_pt: torch.Tensor) -> torch.Tensor:
        """
        Repeats the polar map on a model with dropout to get a point cloud prediction.

        Args:
            batch_polar_pt: tensor corresponding to a batch of 3D polar maps.

        Returns:
            batch of prediction of size (batch_size, n_repeats, 2, n_angles)
        """
        self.model.set_mode("dropout" if self.dropout else "eval")
        batch_size, dn_angles = batch_polar_pt.shape[0], batch_polar_pt.shape[-2]
        if self.dimension == 3:
            repeat_polar_pt = batch_polar_pt.repeat(self.n_repeats, 1, 1, 1, 1)
        else:
            repeat_polar_pt = batch_polar_pt.repeat(self.n_repeats, 1, 1, 1)

        with torch.no_grad():
            batch_prediction_pt = (
                self.model(repeat_polar_pt.to(self.device))
                .cpu()
                .reshape(self.n_repeats, batch_size, 2, dn_angles // 2 + 1)
            )

        return batch_prediction_pt.transpose(0, 1)

    @staticmethod
    def interpolate_contour(
        contour_pt: torch.Tensor, n_angles: int, delta_theta: float
    ) -> torch.Tensor:
        """
        Takes a set of points in one unique slice in cartesian coordinates and smooth it to extract one
        contour point per angle. A new center is estimated (barycenter of all points)

        Args:
            contour_pt: noisy point cloud around the contour.
            n_angles: number of points to extract.
            delta_theta: theta range used to compute uncertainty.

        Returns:
            smooth contour of n_angles points in cartesian coordinates.
        """
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline

        center_pt = torch.mean(contour_pt, dim=0)
        polar_pt = cart2polar(contour_pt, center_pt)

        # Interpolate on sine / cosine to ensure periodicity
        x_train = torch.stack(
            (torch.cos(polar_pt[:, 2]), torch.sin(polar_pt[:, 2])), dim=1
        ).numpy()
        y_train = polar_pt[:, 1].numpy().reshape(-1, 1)

        # Interpolation between ray and angle
        model = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-3, random_state=42))
        model.fit(x_train, y_train)

        # Cover the whole domain ]-pi; pi[
        angles = np.linspace(-np.pi, np.pi, num=n_angles, endpoint=False)
        x_interp = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        rays = model.predict(x_interp)[:, 0]

        # Compute uncertainty of each new point
        local_std_pt = torch.zeros(len(angles))
        for idx_angle, angle in enumerate(angles):
            min_angle = angle - delta_theta / 2
            max_angle = angle + delta_theta / 2

            # Find which points in original data should be used in the vicinity of the interpolated one
            local_indices = (polar_pt[:, 2] < max_angle) & (polar_pt[:, 2] > min_angle)
            if angle < polar_pt[:, 2].min():
                local_indices = local_indices | (
                    (polar_pt[:, 2] < max_angle + 2 * np.pi)
                    & (polar_pt[:, 2] > min_angle + 2 * np.pi)
                )
            elif angle > polar_pt[:, 2].max():
                local_indices = local_indices | (
                    (polar_pt[:, 2] < max_angle - 2 * np.pi)
                    & (polar_pt[:, 2] > min_angle - 2 * np.pi)
                )

            x_local = x_train[local_indices]
            y_local = polar_pt[:, 1][local_indices].numpy()
            estimated_local = model.predict(x_local)[:, 0]
            local_std = np.sum((y_local - estimated_local) ** 2) / len(y_local)
            local_std_pt[idx_angle] = local_std

        interp_polar_pt = torch.zeros(n_angles, 3)
        slice_idx = contour_pt[:, 0].unique().item()
        interp_polar_pt[:, 0] = slice_idx
        interp_polar_pt[:, 1] = torch.from_numpy(rays)
        interp_polar_pt[:, 2] = torch.from_numpy(angles)

        # Normalize deviation according to mean ray
        local_norm_std = local_std_pt / interp_polar_pt[:, 1]

        output_pt = polar2cart(interp_polar_pt, center_pt)
        output_pt = torch.hstack(
            (output_pt, local_std_pt.reshape(-1, 1), local_norm_std.reshape(-1, 1))
        )
        return output_pt


class ContourNet(nn.Module):
    def __init__(
        self, hidden_size: int = 16, dropout: bool = False, dimension: int = 3
    ):
        super(ContourNet, self).__init__()
        self.dropout_value = 0.2
        self.dimension = dimension

        self.conv_layer = nn.Conv3d if self.dimension == 3 else nn.Conv2d
        self.norm_layer = nn.BatchNorm3d if self.dimension == 3 else nn.BatchNorm2d

        model = [
            self.conv_layer(1, hidden_size, kernel_size=3),
            self.norm_layer(hidden_size),
            nn.ReLU(),
        ]
        max_pow_z = 2
        max_pow_a = 4
        # Dilation_rate
        for dr in range(1, max_pow_z):
            model += self.compute_module_list(
                in_channels=dr * hidden_size,
                out_channels=(dr + 1) * hidden_size,
                kernel_size=3,
                dilation=2**dr,
                dropout=dropout,
            )

        for dr in range(max_pow_z, max_pow_a):
            model += self.compute_module_list(
                in_channels=dr * hidden_size,
                out_channels=(dr + 1) * hidden_size,
                kernel_size=(1, 3, 3) if dimension == 3 else 3,
                dilation=2**dr,
                dropout=dropout,
            )

        max_pow = 6
        for dr in range(max_pow_a, max_pow):
            model += self.compute_module_list(
                in_channels=dr * hidden_size,
                out_channels=(dr + 1) * hidden_size,
                kernel_size=(1, 1, 3) if dimension == 3 else (1, 3),
                dilation=(1, 1, 2**dr) if dimension == 3 else (1, 2**dr),
                dropout=dropout,
            )

        model += self.compute_module_list(
            in_channels=max_pow * hidden_size,
            out_channels=max_pow * hidden_size,
            kernel_size=1,
            dilation=1,
            dropout=dropout,
        )
        model.append(self.conv_layer(max_pow * hidden_size, 2, kernel_size=1))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x.squeeze(-3).squeeze(-1)

    def compute_module_list(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int, int], int],
        dilation: Union[Tuple[int, int, int], int],
        dropout: float,
    ) -> List[nn.Module]:

        module_list = [
            self.conv_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        ]
        if dropout:
            module_list.append(nn.Dropout(p=self.dropout_value))
        module_list += [self.norm_layer(out_channels), nn.ReLU()]

        return module_list

    def set_mode(self, mode="eval"):
        if mode == "eval":
            return self.model.eval()
        elif mode == "train":
            return self.model.train()
        elif mode == "dropout":
            self.model.eval()
            for m in self.model.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()

            return self.model
