from typing import Callable
from torch.autograd.function import NestedIOFunction
from torchseal.wrapper import CKKSWrapper


class CKKSLinearFunctionWrapper(NestedIOFunction):
    enc_input: CKKSWrapper


class CKKSPoolingFunctionWrapper(NestedIOFunction):
    pass


class CKKSLossFunctionWrapper(NestedIOFunction):
    enc_output: CKKSWrapper
    enc_target: CKKSWrapper


class CKKSActivationFunctionWrapper(NestedIOFunction):
    enc_input: CKKSWrapper
    polyval_derivative: Callable[[float], float]


class CKKSSoftmaxFunctionWrapper(NestedIOFunction):
    enc_output: CKKSWrapper
