from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSSoftmaxFunctionWrapper

import typing
import numpy as np
import torch


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSSoftmaxFunctionWrapper, enc_input: CKKSWrapper, exp_coeffs: np.ndarray, inverse_coeffs: np.ndarray, inverse_iterations: int) -> CKKSWrapper:
        # Apply the division function to the encrypted input
        enc_output = enc_input.do_softmax(
            exp_coeffs, inverse_coeffs, inverse_iterations
        )

        # Save the ctx for the backward method
        ctx.enc_output = enc_output.clone()

        return enc_output

    @staticmethod
    def backward(ctx: CKKSSoftmaxFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None, None]:
        # Get the saved tensors
        enc_output = ctx.enc_output.do_decryption()

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[bool, bool, bool, bool],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        grad_input = None

        if result[0]:
            # Compute the gradients
            grad_input = enc_output * \
                (grad_output - (grad_output * enc_output).sum(dim=1, keepdim=True))

        return grad_input, None, None, None
