from typing import Dict, Any, Tuple, List, Union
from carotid.utils.transforms import polar2cart, cart2polar
import pandas as pd
from torch import nn
import torch
import numpy as np
from os import path, listdir
from tqdm import tqdm


class ContourTransform:
    def __init__(self, parameters: Dict[str, Any]):
        self.model_dir = parameters["model_dir"]
        self.device = parameters["device"]
        self.eval_mode = parameters["eval_mode"]
        self.n_repeats = parameters["n_repeats"]
        self.side_list = ["left", "right"]

        self.model_paths_list = [
            path.join(self.model_dir, filename)
            for filename in listdir(self.model_dir)
            if filename.endswith(".pt")
        ]
        self.model = CONV3D(dropout=0.2 if self.eval_mode == "dropout" else 0).to(
            self.device
        )
        self.model.load_state_dict(
            torch.load(self.model_paths_list[0], map_location=self.device)
        )

    def __call__(self, sample):
        for side in self.side_list:
            contour_df = pd.DataFrame(columns=["label", "object", "z", "y", "x"])
            polar_list = sample[f"{side}_polar"]
            orig_shape = sample[f"{side}_polar_meta_dict"]["orig_shape"]
            affine = sample[f"{side}_polar_meta_dict"]["affine"]
            for polar_dict in polar_list:
                label = polar_dict["label"]
                lumen_cont, wall_cont = self._transform(
                    polar_dict["polar_pt"],
                    polar_dict["center_pt"],
                    sample[f"{side}_polar_meta_dict"],
                )

                # Build DataFrames
                lumen_slice_df = pd.DataFrame(
                    data=lumen_cont.numpy(), columns=["z", "y", "x"]
                )
                wall_slice_df = pd.DataFrame(
                    data=wall_cont.numpy(), columns=["z", "y", "x"]
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
        polar_pt: torch.Tensor,
        center_pt: torch.Tensor,
        polar_meta_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_angles = polar_meta_dict["n_angles"]
        polar_ray = polar_meta_dict["polar_ray"]
        cartesian_ray = polar_meta_dict["cartesian_ray"]

        if self.eval_mode == "dropout":
            prediction_pt = self.ensemble_dropout(polar_pt)
        else:
            prediction_pt = self.ensemble_models(polar_pt, n_angles)

        lumen_cloud, wall_cloud = self.pred2cartesian(
            prediction_pt,
            center_pt=center_pt,
            n_angles=n_angles,
            polar_ray=polar_ray,
            cartesian_ray=cartesian_ray,
        )

        lumen_cont = self.interpolate_contour(lumen_cloud, n_angles)
        wall_cont = self.interpolate_contour(wall_cloud, n_angles)

        return lumen_cont, wall_cont

    @staticmethod
    def pred2cartesian(
        prediction_pt: torch.Tensor,
        center_pt: torch.Tensor,
        n_angles: int,
        polar_ray: int,
        cartesian_ray: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps N polar predictions of the CNN to the cartesian space.

        Each polar prediction is of size (2, n_angles):
            - First column corresponds to be the distance between the left border of the image and the lumen.
            - Second column is the wall width.
        Corresponding angles correspond to an equal split of the trigonometrical circle in n_angles.
        """
        n_pred, n_coords, n_angles_pred = prediction_pt.shape
        assert n_coords == 2
        assert n_angles_pred == n_angles

        # First coordinate of prediction is the distance between the border and the lumen
        lumen_dists = (
            (polar_ray / 2 - prediction_pt[:, 0, :]) * cartesian_ray / polar_ray
        )
        # Second coordinate of prediction is the wall width
        wall_dists = lumen_dists + prediction_pt[:, 1, :] * cartesian_ray / polar_ray

        angle_vals = torch.linspace(0, 2 * np.pi, (n_angles + 1))[:n_angles].reshape(
            1, -1
        )
        lumen_cont = torch.zeros((n_pred, n_angles, 3))
        wall_cont = torch.zeros_like(lumen_cont)
        lumen_cont[:, :, 1] = lumen_dists * torch.cos(angle_vals) + center_pt[1]
        lumen_cont[:, :, 2] = lumen_dists * torch.sin(angle_vals) + center_pt[2]
        wall_cont[:, :, 1] = wall_dists * torch.cos(angle_vals) + center_pt[1]
        wall_cont[:, :, 2] = wall_dists * torch.sin(angle_vals) + center_pt[2]

        lumen_cont = lumen_cont.reshape((n_pred * n_angles, 3))
        wall_cont = wall_cont.reshape((n_pred * n_angles, 3))

        # Report slice index
        lumen_cont[:, 0] = center_pt[0]
        wall_cont[:, 0] = center_pt[0]

        return lumen_cont, wall_cont

    def ensemble_models(self, polar_pt: torch.Tensor, n_angles: int) -> torch.Tensor:
        all_pred_pt = torch.zeros((len(self.model_paths_list), 2, n_angles))

        for model_index, model_path in tqdm(
            enumerate(self.model_paths_list), desc="Predicting heatmaps", leave=False
        ):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.set_mode("eval")

            with torch.no_grad():
                all_pred_pt[model_index] = (
                    self.model(polar_pt.unsqueeze(0).to(self.device)).squeeze(0).cpu()
                )

        return all_pred_pt

    def ensemble_dropout(self, polar_pt: torch.Tensor) -> torch.Tensor:
        """
        Repeats the polar map on a model with dropout to get a point cloud prediction.

        Args:
            polar_pt: tensor corresponding to one single 3D polar map.

        Returns:
            Values of the prediction
                - first dimension corresponds to the number of contours predicted,
                - second dimension differentiate the objects (lumen or wall),
                - third dimension is the number of angles per prediction.
        """
        self.model.set_mode("dropout")

        repeat_polar_pt = polar_pt.repeat(self.n_repeats, 1, 1, 1, 1)
        with torch.no_grad():
            all_pred_pt = self.model(repeat_polar_pt).cpu()

        return all_pred_pt

    @staticmethod
    def interpolate_contour(contour_pt: torch.Tensor, n_angles: int) -> torch.Tensor:
        """
        Takes a set of points in one unique slice in cartesian coordinates and smooth it to extract one contour point per angle.
        A new center is estimated (barycenter of all points)

        Args:
            contour_pt: noisy point cloud around the contour
            n_angles: number of points to extract

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

        interp_polar_pt = torch.zeros(n_angles, 3)
        slice_idx = contour_pt[:, 0].unique().item()
        interp_polar_pt[:, 0] = slice_idx
        interp_polar_pt[:, 1] = torch.from_numpy(rays)
        interp_polar_pt[:, 2] = torch.from_numpy(angles)

        output_pt = polar2cart(interp_polar_pt, center_pt)

        return output_pt


class CONV3D(nn.Module):
    def __init__(self, hidden_size: int = 16, dropout: float = 0.2):
        super(CONV3D, self).__init__()
        model = [
            nn.Conv3d(1, hidden_size, kernel_size=3),
            nn.BatchNorm3d(hidden_size),
            nn.ReLU(),
        ]
        max_pow_z = 2
        max_pow_a = 4
        # Dilation_rate
        for dr in range(1, max_pow_z):
            model += self.compute_module_list(
                in_channels=dr * hidden_size,
                out_channels=(dr + 1) * hidden_size,
                kernel_size=(3, 3, 3),
                dilation=2**dr,
                dropout=dropout,
            )

        for dr in range(max_pow_z, max_pow_a):
            model += self.compute_module_list(
                in_channels=dr * hidden_size,
                out_channels=(dr + 1) * hidden_size,
                kernel_size=(1, 3, 3),
                dilation=2**dr,
                dropout=dropout,
            )

        max_pow = 6
        for dr in range(max_pow_a, max_pow):
            model += self.compute_module_list(
                in_channels=dr * hidden_size,
                out_channels=(dr + 1) * hidden_size,
                kernel_size=(1, 1, 3),
                dilation=(1, 1, 2**dr),
                dropout=dropout,
            )

        model += self.compute_module_list(
            in_channels=max_pow * hidden_size,
            out_channels=max_pow * hidden_size,
            kernel_size=(1, 1, 1),
            dilation=1,
            dropout=dropout,
        )
        model.append(nn.Conv3d(max_pow * hidden_size, 2, kernel_size=(1, 1, 1)))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x.squeeze(-3).squeeze(-1)

    @staticmethod
    def compute_module_list(
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        dilation: Union[Tuple[int, int, int], int],
        dropout: float,
    ) -> List[nn.Module]:
        module_list = [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
        ]
        if 0 < dropout < 1:
            module_list.append(nn.Dropout(p=dropout))
        module_list += [nn.BatchNorm3d(out_channels), nn.ReLU()]

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
