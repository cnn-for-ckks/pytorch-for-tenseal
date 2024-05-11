from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSLinearFunctionWrapper

import typing
import numpy as np
import torch


class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSLinearFunctionWrapper, enc_input: CKKSWrapper) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.enc_input = enc_input.clone()

        # Apply square function to the encrypted input (x ** 2)
        enc_output = enc_input.do_encrypted_square()

        return enc_output

    # TODO: Move this to train mode
    @staticmethod
    def backward(ctx: CKKSLinearFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor]]:
        # Get the saved tensors
        input = ctx.enc_input.do_decryption()

        # Do the backward operation (2 * x)
        backward_output = input.mul(2)

        # Get the needs_input_grad
        result = typing.cast(Tuple[bool], ctx.needs_input_grad)

        # Initialize the gradients
        grad_input = None

        if result[0]:
            # Compute the gradients
            grad_input = grad_output.mul(backward_output)

        return grad_input,
