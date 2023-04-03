import torch


class DeviceError(ValueError):
    pass


def check_device(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device {device} was automatically chosen.")
    else:
        device = torch.device(device)

    test_pt = torch.Tensor([0])
    try:
        test_pt.to(device)
    except (AssertionError, RuntimeError):
        raise DeviceError(
            f"Device {device} could not be found by pytorch."
            f"If this device was automatically found, please set explicitly which device"
            f"should be used."
        )

    return device
