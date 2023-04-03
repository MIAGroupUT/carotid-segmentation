import torch
from carotid.utils.device import check_device, DeviceError
import pytest


def test_check_device():
    # Create a non-GPU environment
    torch.cuda.is_available = lambda: False

    assert check_device() == torch.device("cpu")
    assert check_device("cpu") == torch.device("cpu")
    with pytest.raises(DeviceError):
        check_device("cuda")
