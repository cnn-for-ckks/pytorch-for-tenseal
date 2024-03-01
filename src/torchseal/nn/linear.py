from typing import Optional
from torch import Tensor
from torch.nn import Module, Parameter
from torch.autograd import Function
from torch.autograd.function import NestedIOFunction
from tenseal import CKKSVector

import torch


class LinearFunctionCtx(NestedIOFunction):
    enc_x: CKKSVector


class LinearFunction(Function):
    @staticmethod
    def forward(ctx: LinearFunctionCtx, enc_x: CKKSVector, weight: Tensor, bias: Tensor) -> CKKSVector:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight, bias)
        ctx.enc_x = enc_x

        return enc_x.matmul(weight.tolist()).add(bias.tolist())

    # TODO: Define the backward method to enable training
    @staticmethod
    def backward(ctx: LinearFunctionCtx, enc_grad_output: CKKSVector):
        return super(LinearFunction, LinearFunction).backward(ctx, enc_grad_output)

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

    def __init__(self, in_features: int, out_features: int, weight: Optional[Tensor], bias: Optional[Tensor]):
        super(Linear, self).__init__()

        self.weight = Parameter(
            torch.empty(in_features, out_features, requires_grad=True)
        ) if weight is None else weight
        self.bias = Parameter(
            torch.empty(out_features, requires_grad=True)
        ) if bias is None else bias

    def forward(self, enc_x: CKKSVector) -> CKKSVector:
        return LinearFunction.apply(enc_x, self.weight, self.bias)
