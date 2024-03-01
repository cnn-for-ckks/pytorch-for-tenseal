from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Module, Parameter
from torch.autograd import Function
from torch.autograd.function import NestedIOFunction
from tenseal import CKKSVector

import torch


# NOTE: Unused at the moment
class LinearFunctionCtx(NestedIOFunction):
    enc_x: CKKSVector


# NOTE: Unused at the moment
class LinearFunction(Function):
    @staticmethod
    def forward(ctx: LinearFunctionCtx, enc_x: CKKSVector, weight: Tensor, bias: Tensor) -> CKKSVector:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight, bias)
        ctx.enc_x = enc_x

        return enc_x.matmul(weight.tolist()).add(bias.tolist())

    # NOTE: This method requires private key access
    @staticmethod
    def backward(ctx: LinearFunctionCtx, enc_grad_output: CKKSVector) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        # Get the saved tensors
        weight, bias = ctx.saved_tensors
        enc_x = ctx.enc_x

        # Check if the weight and bias have the right type
        if type(weight) != torch.Tensor or type(bias) != torch.Tensor:
            raise TypeError("The weight and bias should be tensors")

        # Get the input gradient
        result: Tuple[bool, bool, bool] = ctx.needs_input_grad  # type: ignore

        # Decrypt the encrypted input and gradient
        x = torch.tensor(enc_x.decrypt(), requires_grad=True)
        grad_output = torch.tensor(
            enc_grad_output.decrypt(),
            requires_grad=True
        )

        # Initialize the gradients
        grad_input = grad_weight = grad_bias = None

        # Compute the gradients
        if result[0]:
            grad_input = grad_output.mm(weight)
        if result[1]:
            grad_weight = grad_output.t().mm(x)
        if result[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

    @staticmethod
    def apply(enc_x: CKKSVector, weight: Tensor, bias: Tensor) -> CKKSVector:
        result = super(LinearFunction, LinearFunction).apply(
            enc_x, weight, bias
        )

        if type(result) != CKKSVector:
            raise TypeError("The result should be a CKKSVector")

        return result


class Linear(Module):
    weight: Tensor
    bias: Tensor

    def __init__(self, in_features: int, out_features: int, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None):
        super(Linear, self).__init__()

        self.weight = Parameter(
            torch.rand(in_features, out_features),
            requires_grad=True
        ) if weight is None else weight
        self.bias = Parameter(
            torch.rand(out_features),
            requires_grad=True
        ) if bias is None else bias

    def forward(self, enc_x: CKKSVector) -> CKKSVector:
        return LinearFunction.apply(enc_x, self.weight, self.bias)
