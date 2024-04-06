from typing import Callable
from torch.autograd.function import NestedIOFunction
from torchseal.wrapper.ckks import CKKSWrapper

import torch
import numpy as np


class CKKSFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper


class CKKSConvFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper
    output_size: torch.Size
    stride: int
    padding: int


class CKKSActivationFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper
    polyval_derivative: Callable[[float], float]
