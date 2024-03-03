from typing import Tuple
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import NestedIOFunction
from tenseal import CKKSVector

import torch


class SigmoidFunctionCtx(NestedIOFunction):
    enc_x: CKKSVector


class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx: SigmoidFunctionCtx, enc_x: CKKSVector) -> CKKSVector:
        # Save the ctx for the backward method
        ctx.enc_x = enc_x

        # TODO: Do approximation of the sigmoid function using the polynomial approximation

        # Apply the polynomial approximation to the encrypted input
        # NOTE: This is just an example
        result: CKKSVector = enc_x.polyval(
            [0.5, 0.197, 0, -0.004]
        )  # type: ignore

        return result

    @staticmethod
    def backward(ctx: SigmoidFunctionCtx, enc_grad_output: CKKSVector) -> Tuple[Tensor]:
        # Get the saved tensors
        enc_x = ctx.enc_x

        # Decrypt the encrypted input and gradient
        x = torch.tensor(enc_x.decrypt(), requires_grad=True)
        grad_output = torch.tensor(
            enc_grad_output.decrypt(), requires_grad=True
        )

        # TODO: Do approximation of the sigmoid function using the polynomial approximation

        # Compute the gradients
        # NOTE: This is just an example
        grad_input = grad_output.mm(x.apply_(lambda x: 0.197 - 0.008 * x))

        return grad_input,

    @staticmethod
    def apply(enc_x: CKKSVector) -> CKKSVector:
        result: CKKSVector = super(
            SigmoidFunction, SigmoidFunction
        ).apply(enc_x)  # type: ignore

        return result
