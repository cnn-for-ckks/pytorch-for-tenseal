from typing import Tuple
from torch.autograd.function import NestedIOFunction
from torchseal.wrapper.ckks import CKKSWrapper


class CKKSFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper


class CKKSConvFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper
    output_size: Tuple[int, int]
    kernel_size: Tuple[int, int]
