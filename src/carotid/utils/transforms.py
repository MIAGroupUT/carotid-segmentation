import monai
from monai.config import KeysCollection
from monai.utils.enums import TraceKeys
import numpy as np
from copy import deepcopy


class ExtractLeftAndRightd(monai.transforms.InvertibleTransform):
    """
    Extract the left (first two thirds) and right (last two thirds) parts of an image.
    The right part is flipped.
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
                if self.trace_key(key) in d:
                    d[self.trace_key(f"left_{key}")] = deepcopy(d[self.trace_key(key)])
                    d[self.trace_key(f"right_{key}")] = deepcopy(d[self.trace_key(key)])
                    del d[self.trace_key(key)]

            self.push_transform(
                d,
                f"left_{key}",
                extra_info={"side": "left"},
                orig_size=d[left_key].shape[1::],
            )
            self.push_transform(
                d,
                f"right_{key}",
                extra_info={"side": "right"},
                orig_size=d[right_key].shape[1::],
            )

            d[f"left_{key}"] = deepcopy(self.extract_left(d[left_key]))
            d[f"right_{key}"] = deepcopy(self.extract_right(d[right_key]))

            if key in d:
                del d[key]

            d[f"left_{key}_{self.meta_key_postfix}"] = deepcopy(
                d[f"{left_key}_{self.meta_key_postfix}"]
            )
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
                    left_np=d[f"left_{key}"], right_np=d[f"right_{key}"]
                )
                d[self.trace_key(key)] = deepcopy(d[self.trace_key(f"left_{key}")])
                d[f"{key}_{self.meta_key_postfix}"] = deepcopy(
                    d[f"left_{key}_{self.meta_key_postfix}"]
                )
                # After combination forget the previous separated versions
                for side in ["left", "right"]:
                    del d[f"{side}_{key}"]
                    del d[f"{side}_{key}_{self.meta_key_postfix}"]
                    del d[self.trace_key(f"{side}_key")]

        return d

    def _run_inversion(self, d, key):
        transform = self.get_most_recent_transform(d, key)
        orig_size = transform[TraceKeys.ORIG_SIZE]
        current_size = d[key].shape
        side = transform[TraceKeys.EXTRA_INFO]["side"]
        padding_size = orig_size[-1] - current_size[-1]
        pad_transform = monai.transforms.Pad(
            [(0, 0), (0, 0), (0, 0), (padding_size, 0)]
        )

        d[key] = pad_transform(d[key])

        if side == "right":
            d[key] = np.flip(d[key], axis=-1)

        self.pop_transform(d, key)

        return d

    @staticmethod
    def recombine(left_np: np.ndarray, right_np: np.ndarray) -> np.ndarray:
        orig_y = left_np.shape[-1]
        recombined_np = np.zeros_like(left_np)
        recombined_np[..., : orig_y // 3 :] = right_np[..., : orig_y // 3 :]
        recombined_np[..., orig_y * 2 // 3 : :] = left_np[..., orig_y * 2 // 3 : :]
        recombined_np[..., orig_y // 3 : orig_y * 2 // 3 :] = (
            right_np[..., orig_y // 3 : orig_y * 2 // 3 :]
            + left_np[..., orig_y // 3 : orig_y * 2 // 3 :]
        ) / 2
        return recombined_np

    @staticmethod
    def extract_left(image_np: np.ndarray) -> np.ndarray:
        """Extract the first two thirds of the image."""
        return image_np[..., image_np.shape[-1] // 3 : :]

    @staticmethod
    def extract_right(image_np: np.ndarray) -> np.ndarray:
        """Extract the last two thirds of the image and flip it."""
        return np.flip(image_np[..., 0 : 2 * image_np.shape[-1] // 3], axis=-1)


class BuildEmptyLabelsd(monai.transforms.Transform):
    """
    Create left and right empty labels corresponding to the image.
    """

    def __init__(
        self,
        image_key: str = "img",
        meta_key_postfix: str = "meta_dict",
    ):
        self.image_key = image_key
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        d = dict(data)

        meta_dict = data[f"{self.image_key}_{self.meta_key_postfix}"]
        orig_size = data[f"{self.image_key}"].shape[1::]

        for side in ["left", "right"]:
            d[f"{side}_label"] = np.zeros([2, *orig_size])
            d[f"{side}_label_meta_dict"] = meta_dict.copy()
        return d
