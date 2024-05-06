from typing import Callable, Tuple
from torch.autograd.function import NestedIOFunction
from torchseal.wrapper import CKKSWrapper


class CKKSFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper


class CKKSActivationFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper
    polyval_derivative: Callable[[float], float]


class CKKSSoftmaxFunctionWrapper(NestedIOFunction):
    out_x: CKKSWrapper


class CKKSConvFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper
    input_size_with_channel: Tuple[int, int, int, int]
    stride: int
    padding: int
