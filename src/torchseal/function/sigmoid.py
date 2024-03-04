from typing import Tuple
from torch import Tensor
from torch.autograd import Function
from tenseal import CKKSVector
from torchseal.wrapper.function import CKKSFunctionCtx

import torch


class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx: CKKSFunctionCtx, enc_x: CKKSVector) -> CKKSVector:
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
    def backward(ctx: CKKSFunctionCtx, grad_output: Tensor) -> Tuple[Tensor]:
        # Get the saved tensors
        enc_x = ctx.enc_x

        # Decrypt the encrypted input and gradient
        out_x = torch.tensor(
            list(map(lambda x: 0.197 - 0.008 * x, enc_x.decrypt())),
            requires_grad=True
        )

        # TODO: Do approximation of the sigmoid function using the polynomial approximation

        # Compute the gradients
        # NOTE: This is just an example
        grad_input = grad_output.mm(out_x)

        return grad_input,
