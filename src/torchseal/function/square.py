from typing import Tuple
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import NestedIOFunction
from tenseal import CKKSVector

import torch


class SquareFunctionCtx(NestedIOFunction):
    enc_x: CKKSVector


class SquareFunction(Function):
    @staticmethod
    def forward(ctx: SquareFunctionCtx, enc_x: CKKSVector) -> CKKSVector:
        # Save the ctx for the backward method
        ctx.enc_x = enc_x

        # Apply square function to the encrypted input
        result: CKKSVector = enc_x.square()  # type: ignore

        return result

    @staticmethod
    def backward(ctx: SquareFunctionCtx, enc_grad_output: CKKSVector) -> Tuple[Tensor]:
        # Get the saved tensors
        enc_x = ctx.enc_x

        # Decrypt the encrypted input and gradient
        x = torch.tensor(enc_x.decrypt(), requires_grad=True)
        grad_output = torch.tensor(
            enc_grad_output.decrypt(), requires_grad=True
        )

        # Compute the gradients
        grad_input = grad_output.mm(x.apply_(lambda x: 2 * x))

        return grad_input,

    @staticmethod
    def apply(enc_x: CKKSVector) -> CKKSVector:
        result: CKKSVector = super(
            SquareFunction, SquareFunction
        ).apply(enc_x)  # type: ignore

        return result
