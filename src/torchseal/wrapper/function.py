from torch.autograd.function import NestedIOFunction
from torchseal.wrapper.ckks import CKKSWrapper

import torch


class CKKSFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper


class CKKSConvFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper
    output_size: torch.Size
    stride: int
    padding: int
