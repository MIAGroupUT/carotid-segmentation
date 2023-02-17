import monai
from monai.config import KeysCollection
from monai.utils.enums import TraceKeys, PytorchPadMode
from monai.data.meta_tensor import MetaTensor, get_track_meta
import torch
from copy import deepcopy

from torch import Tensor


# TODO: work on non-inverted images
class ExtractLeftAndRightd(monai.transforms.InvertibleTransform):
    """
    Extract the left (first two thirds) and right (last two thirds) parts of an image.
    The right part is flipped.
    Assumes that the first dimension is the channel axis and that the orientation is SPL.
    """

    def __init__(
        self,
        keys: KeysCollection,
        meta_key_postfix: str = "meta_dict",
        split_keys: KeysCollection = None,
    ):
        """
        Args:
            keys: is used to get and set the actual data
                item to transform.
            meta_key_postfix: use `key_{postfix}` to fetch the meta-data according
                to the key data, default is `meta_dict`, the meta-data is a dictionary object.
                For example, to handle key `image`,  read/write affine matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.
            split_keys: list of keys for which the 'left' and 'right' volumes are already available.
        """
        self.keys = keys
        self.meta_key_postfix = meta_key_postfix
        self.split_keys = split_keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if self.split_keys is not None and key in self.split_keys:
                left_key = f"left_{key}"
                right_key = f"right_{key}"
            else:
                left_key, right_key = key, key

            left_orig_size = d[left_key].shape[1::]
            right_orig_size = d[right_key].shape[1::]

            d[f"left_{key}"] = deepcopy(self.extract_left(d[left_key]))
            d[f"right_{key}"] = deepcopy(self.extract_right(d[right_key]))

            if get_track_meta():
                self.push_transform(
                    d,
                    f"left_{key}",
                    extra_info={"side": "left"},
                    orig_size=left_orig_size,
                )
                self.push_transform(
                    d,
                    f"right_{key}",
                    extra_info={"side": "right"},
                    orig_size=right_orig_size,
                )

            if key in d:
                del d[key]

            if f"{left_key}_{self.meta_key_postfix}" in d:
                d[f"left_{key}_{self.meta_key_postfix}"] = deepcopy(
                    d[f"{left_key}_{self.meta_key_postfix}"]
                )
            if f"{right_key}_{self.meta_key_postfix}" in d:
                d[f"right_{key}_{self.meta_key_postfix}"] = deepcopy(
                    d[f"{right_key}_{self.meta_key_postfix}"]
                )

            if f"{key}_{self.meta_key_postfix}" in d:
                del d[f"{key}_{self.meta_key_postfix}"]

        return d

    def inverse(self, data):
        d = deepcopy(dict(data))
        for key in self.keys:
            for side in ["right", "left"]:
                d = self._run_inversion(d, f"{side}_{key}")
            if self.split_keys is None or key not in self.split_keys:
                d[key] = self.recombine(
                    left_pt=d[f"left_{key}"], right_pt=d[f"right_{key}"]
                )

                # After combination forget the previous separated versions
                for side in ["left", "right"]:
                    del d[f"{side}_{key}"]
                    if f"{side}_{key}_{self.meta_key_postfix}" in d:
                        del d[f"{side}_{key}_{self.meta_key_postfix}"]

        return d

    def _run_inversion(self, d, key):
        transform = self.get_most_recent_transform(d, key, pop=True, check=False)
        orig_size = transform[TraceKeys.ORIG_SIZE]
        current_size = d[key].shape
        side = transform[TraceKeys.EXTRA_INFO]["side"]
        padding_size = orig_size[-1] - current_size[-1]
        to_pad = [(0, 0), (0, 0), (0, 0), (padding_size, 0)]
        pad_transform = monai.transforms.Pad()

        d[key] = pad_transform._pt_pad(d[key], to_pad, mode=PytorchPadMode.CONSTANT)

        if side == "right":
            d[key] = torch.flip(d[key], dims=(-1,))

        return d

    @staticmethod
    def recombine(left_pt: MetaTensor, right_pt: MetaTensor) -> MetaTensor:
        orig_y = left_pt.shape[-1]
        recombined_pt = left_pt.clone()
        recombined_pt[:] = 0
        recombined_pt[..., : orig_y // 3 :] = right_pt[..., : orig_y // 3 :]
        recombined_pt[..., orig_y * 2 // 3 : :] = left_pt[..., orig_y * 2 // 3 : :]
        recombined_pt[..., orig_y // 3 : orig_y * 2 // 3 :] = (
            right_pt[..., orig_y // 3 : orig_y * 2 // 3 :]
            + left_pt[..., orig_y // 3 : orig_y * 2 // 3 :]
        ) / 2
        return recombined_pt

    @staticmethod
    def extract_left(image_pt: Tensor) -> Tensor:
        """Extract the first two thirds of the image and flip it."""
        return image_pt[..., image_pt.shape[-1] // 3 : :]

    @staticmethod
    def extract_right(image_pt: Tensor) -> Tensor:
        """Extract the last two thirds of the image."""
        return torch.flip(image_pt[..., 0 : 2 * image_pt.shape[-1] // 3], dims=(-1,))


class BuildEmptyHeatmapd(monai.transforms.Transform):
    """
    Create left and right empty labels corresponding to the image.
    """

    def __init__(
        self,
        image_key: str = "image",
        meta_key_postfix: str = "meta_dict",
    ):
        self.image_key = image_key
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        d = dict(data)

        zero_meta_tensor = data[f"{self.image_key}"].repeat((2, 1, 1, 1))
        zero_meta_tensor[:] = 0

        for side in ["left", "right"]:
            d[f"{side}_heatmap"] = zero_meta_tensor.clone()
        return d


class Printd(monai.transforms.InvertibleTransform, monai.transforms.MapTransform):
    def __init__(self, keys, allow_missing_keys=False, message=""):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.message = message

    def __call__(self, d):
        self.print(d, inverse=False)
        return d

    def inverse(self, d):
        self.print(d, inverse=True)
        return d

    def print(self, d, inverse=False):
        for key in self.key_iterator(d):
            print(self.message)
            if inverse:
                print("Printing inverse", key)
            else:
                print("Printing", key)

            print("Type", type(d[key]))
            if isinstance(d[key], str):
                print((d[key]))
            elif isinstance(d[key], MetaTensor):
                print("Shape", d[key].shape)
                print("Affine", d[key].affine)
                print("Transforms", d[key].applied_operations)

            print()


def cart2polar(cart_pt: torch.Tensor, center_pt: torch.Tensor) -> torch.Tensor:
    """
    Transforms 3D tensor in cartesian coordinates (Z, Y, X) with equal Z to polar (Z, r, theta)
    according to the given center.
    """
    polar_pt = torch.zeros_like(cart_pt)
    # Report Z axis
    polar_pt[:, 0] = cart_pt[:, 0]
    # Find polar ray
    polar_pt[:, 1] = torch.cdist(cart_pt, center_pt.unsqueeze(0))[:, 0]
    # Find angle
    orientation_pt = cart_pt - center_pt.unsqueeze(0)
    polar_pt[:, 2] = torch.arctan2(orientation_pt[:, 2], orientation_pt[:, 1])

    return polar_pt


def polar2cart(polar_pt: torch.Tensor, center_pt: torch.Tensor) -> torch.Tensor:
    """
    Transforms 3D tensor in polar coordinates (Z, r, theta) with equal Z to cartesian (Z, Y, X)
    according to the given center.
    """

    cart_pt = torch.zeros_like(polar_pt)
    # Report Z axis
    cart_pt[:, 0] = polar_pt[:, 0]
    # Compute other Y and X
    cart_pt[:, 1] = polar_pt[:, 1] * torch.cos(polar_pt[:, 2]) + center_pt[1]
    cart_pt[:, 2] = polar_pt[:, 1] * torch.sin(polar_pt[:, 2]) + center_pt[2]

    return cart_pt
