from typing import Dict, Any, Tuple
from torch import nn
import torch
import numpy as np
from os import path, listdir
from tqdm import tqdm


class SegmentationTransform:
    def __init__(self, parameters: Dict[str, Any]):
        self.model_dir = parameters["model_dir"]
        self.device = parameters["device"]
        self.side_list = ["left", "right"]

        self.model_paths_list = [
            path.join(self.model_dir, filename)
            for filename in listdir(self.model_dir)
            if filename.endswith(".pt")
        ]
        self.model = CONV3D().to(self.device)

    def __call__(self, sample):

        for side in self.side_list:
            polar_list = sample[f"{side}_polar"]
            point_cloud = np.zeros((2, 0, 3))
            for polar_dict in polar_list:
                lumen_cont, wall_cont = self._transform(
                    polar_dict["polar_pt"], sample[f"{side}_polar_meta_dict"]
                )
                slice_point_cloud = np.stack((lumen_cont, wall_cont))
                center_np = polar_dict["center"]
                slice_point_cloud += center_np
                point_cloud = np.concatenate((point_cloud, slice_point_cloud), axis=1)
            sample[f"{side}_segmentation"] = point_cloud

        return sample

    def _transform(
        self, polar_pt: torch.Tensor, polar_meta_dict: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_angles = polar_meta_dict["n_angles"]
        polar_ray = polar_meta_dict["polar_ray"]
        cartesian_ray = polar_meta_dict["cartesian_ray"]

        all_pred_pt = torch.zeros((len(self.model_paths_list), 2, n_angles))
        for index, model_path in tqdm(
            enumerate(self.model_paths_list), desc="Predicting heatmaps", leave=False
        ):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

            with torch.no_grad():
                all_pred_pt[index] = (
                    self.model(polar_pt.unsqueeze(0).to(self.device)).squeeze(0).cpu()
                )

        prediction_pt, _ = torch.median(all_pred_pt, dim=0)
        prediction_np = prediction_pt.numpy()
        lumen_cont, wall_cont = self.polar2cartesian(
            prediction_np,
            n_angles=n_angles,
            polar_ray=polar_ray,
            cartesian_ray=cartesian_ray,
        )

        return lumen_cont, wall_cont

    @staticmethod
    def polar2cartesian(
        prediction_np: np.ndarray, n_angles: int, polar_ray: int, cartesian_ray: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Maps polar predictions to the cartesian space"""
        # First coordinate of prediction is the distance between the border and the lumen
        lumen_dists = (polar_ray / 2 - prediction_np[0, :]) * cartesian_ray / polar_ray
        # Second coordinate of prediction is the wall width
        wall_dists = (lumen_dists + prediction_np[1, :]) * cartesian_ray / polar_ray

        angle_vals = np.linspace(0, 2 * np.pi, (n_angles + 1))[:n_angles].reshape(1, -1)
        lumen_cont = np.zeros((n_angles, 3))
        wall_cont = np.zeros_like(lumen_cont)
        lumen_cont[:, 1] = lumen_dists * np.cos(angle_vals)
        lumen_cont[:, 2] = lumen_dists * np.sin(angle_vals)
        wall_cont[:, 1] = wall_dists * np.cos(angle_vals)
        wall_cont[:, 2] = wall_dists * np.sin(angle_vals)
        return lumen_cont, wall_cont


class CONV3D(nn.Module):
    def __init__(self, hidden_size=16):
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
            model += [
                nn.Conv3d(
                    dr * hidden_size,
                    (dr + 1) * hidden_size,
                    kernel_size=(3, 3, 3),
                    dilation=2**dr,
                ),
                nn.BatchNorm3d((dr + 1) * hidden_size),
                nn.ReLU(),
            ]

        for dr in range(max_pow_z, max_pow_a):
            model += [
                nn.Conv3d(
                    dr * hidden_size,
                    (dr + 1) * hidden_size,
                    kernel_size=(1, 3, 3),
                    dilation=2**dr,
                ),
                nn.BatchNorm3d((dr + 1) * hidden_size),
                nn.ReLU(),
            ]

        max_pow = 6
        for dr in range(max_pow_a, max_pow):
            model += [
                nn.Conv3d(
                    dr * hidden_size,
                    (dr + 1) * hidden_size,
                    kernel_size=(1, 1, 3),
                    dilation=(1, 1, 2**dr),
                ),
                nn.BatchNorm3d((dr + 1) * hidden_size),
                nn.ReLU(),
            ]

        model += [
            nn.Conv3d(
                max_pow * hidden_size, max_pow * hidden_size, kernel_size=(1, 1, 1)
            ),
            nn.BatchNorm3d(max_pow * hidden_size),
            nn.ReLU(),
            nn.Conv3d(max_pow * hidden_size, 2, kernel_size=(1, 1, 1)),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x.squeeze(-3).squeeze(-1)
