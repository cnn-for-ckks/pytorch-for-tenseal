from typing import Union
from torch import Tensor
from torch.nn import Module, Parameter
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from tenseal import CKKSVector

import torch


class LinearFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, enc_x: CKKSVector, weight: Tensor, bias: Tensor) -> CKKSVector:
        # TODO: Save the ctx for the backward method

        return enc_x.matmul(weight.tolist()).add(bias.tolist())

    @staticmethod
    def apply(enc_x: CKKSVector, weight: Tensor, bias: Tensor) -> CKKSVector:
        result = super(LinearFunction, LinearFunction).apply(
            enc_x, weight, bias
        )

        if type(result) != CKKSVector:
            raise TypeError("The result should be a CKKSVector")

        return result

    # TODO: Define the backward method to enable training


class Linear(Module):
    weight: Tensor
    bias: Tensor

    def __init__(self, in_features: int, out_features: int, weight: Union[Tensor, None] = None, bias: Union[Tensor, None] = None):
        super(Linear, self).__init__()

        self.weight = Parameter(
            torch.empty(in_features, out_features)
        ) if weight is None else weight
        self.bias = Parameter(
            torch.empty(out_features)
        ) if bias is None else bias

    def forward(self, enc_x: CKKSVector) -> CKKSVector:
        return LinearFunction.apply(enc_x, self.weight, self.bias)
