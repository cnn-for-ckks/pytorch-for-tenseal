from typing import Optional, Tuple
from torch import Tensor
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
        ctx.save_for_backward(weight)
        ctx.enc_x = enc_x

        return enc_x.matmul(weight.tolist()).add(bias.tolist())

    # NOTE: This method requires private key access
    @staticmethod
    def backward(ctx: LinearFunctionCtx, enc_grad_output: CKKSVector) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        # Get the saved tensors
        saved_tensors: Tuple[Tensor] = ctx.saved_tensors  # type: ignore
        enc_x = ctx.enc_x

        # Unpack the saved tensors
        weight, = saved_tensors

        # Get the needs_input_grad
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
            grad_input = grad_output.mm(weight.t())
        if result[1]:
            grad_weight = grad_output.t().mm(x)
        if result[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

    @staticmethod
    def apply(enc_x: CKKSVector, weight: Tensor, bias: Tensor) -> CKKSVector:
        result: CKKSVector = super(LinearFunction, LinearFunction).apply(
            enc_x, weight, bias
        )  # type: ignore

        return result
