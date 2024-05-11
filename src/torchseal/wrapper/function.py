from torch.autograd.function import NestedIOFunction
from torchseal.wrapper import CKKSWrapper

import numpy as np


class CKKSLinearFunctionWrapper(NestedIOFunction):
    enc_input: CKKSWrapper
    training: bool


class CKKSPoolingFunctionWrapper(NestedIOFunction):
    pass


class CKKSLossFunctionWrapper(NestedIOFunction):
    enc_output: CKKSWrapper
    enc_target: CKKSWrapper


class CKKSActivationFunctionWrapper(NestedIOFunction):
    enc_input: CKKSWrapper
    deriv_coeffs: np.ndarray


class CKKSSoftmaxFunctionWrapper(NestedIOFunction):
    enc_output: CKKSWrapper
