from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSLinearFunctionWrapper

import typing
import torch


class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSLinearFunctionWrapper, enc_input: CKKSWrapper) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.enc_input = enc_input.clone()

        # Apply square function to the encrypted input (x ** 2)
        enc_output = enc_input.ckks_encrypted_square()

        return enc_output

    @staticmethod
    def backward(ctx: CKKSLinearFunctionWrapper, enc_grad_output: CKKSWrapper) -> Tuple[Optional[CKKSWrapper]]:
        # Get the saved tensors
        enc_input = ctx.enc_input

        # Get the needs_input_grad
        result = typing.cast(Tuple[bool], ctx.needs_input_grad)

        # Initialize the gradients
        enc_grad_input = None

        if result[0]:
            # Do the backward operation
            enc_backward_output = enc_input.ckks_apply_scalar(2)

            # Compute the gradients
            enc_grad_input = enc_grad_output.ckks_encrypted_apply_mask(
                enc_backward_output
            )

        return enc_grad_input,
