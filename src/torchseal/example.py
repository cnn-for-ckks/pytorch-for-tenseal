import tenseal as ts
import torch
import torchvision


def get_tenseal_version() -> str:
    return ts.__version__


def get_torch_version() -> str:
    return torch.__version__


def get_torchvision_version() -> str:
    return torchvision.__version__
