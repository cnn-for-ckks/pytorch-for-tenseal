from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSActivationFunctionWrapper

import typing
import torch


class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSActivationFunctionWrapper, enc_input: CKKSWrapper) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.enc_input = enc_input.clone()
        ctx.polyval_derivative = lambda x: 2 * x

        # Apply square function to the encrypted input
        enc_output = enc_input.do_square()

        return enc_output

    @staticmethod
    def backward(ctx: CKKSActivationFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor]]:
        # Get the saved tensors
        input = ctx.enc_input.do_decryption()
        polyval_derivative = ctx.polyval_derivative

        # Do the backward operation
        backward_output = input.do_activation_function_backward(
            polyval_derivative
        )

        # Get the needs_input_grad
        result = typing.cast(Tuple[bool], ctx.needs_input_grad)

        # Initialize the gradients
        grad_input = None

        if result[0]:
            # Compute the gradients
            grad_input = grad_output.mul(backward_output)

        return grad_input,
