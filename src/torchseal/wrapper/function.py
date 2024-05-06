from typing import Callable, Tuple
from torch.autograd.function import NestedIOFunction
from torchseal.wrapper import CKKSWrapper


class CKKSFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper


class CKKSPoolingFunctionWrapper(NestedIOFunction):
    pass


class CKKSActivationFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper
    polyval_derivative: Callable[[float], float]


class CKKSSoftmaxFunctionWrapper(NestedIOFunction):
    out_x: CKKSWrapper
