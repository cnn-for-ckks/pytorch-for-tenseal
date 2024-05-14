from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSActivationFunctionWrapper

import typing
import numpy as np
import torch


class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSActivationFunctionWrapper, enc_input: CKKSWrapper, coeffs: np.ndarray, deriv_coeffs: np.ndarray) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.enc_input = enc_input.clone()
        ctx.deriv_coeffs = deriv_coeffs

        # Apply the sigmoid function to the encrypted input
        enc_output = enc_input.ckks_encrypted_polynomial(coeffs)

        return enc_output

    @staticmethod
    def backward(ctx: CKKSActivationFunctionWrapper, enc_grad_output: CKKSWrapper) -> Tuple[Optional[CKKSWrapper], None, None]:
        # Get the saved tensors
        enc_input = ctx.enc_input
        deriv_coeffs = ctx.deriv_coeffs

        # Get the needs_input_grad
        result = typing.cast(Tuple[bool, bool, bool], ctx.needs_input_grad)

        # Initialize the gradients
        enc_grad_input = None

        if result[0]:
            # Do the backward operation
            enc_backward_output = enc_input.ckks_encrypted_polynomial(
                deriv_coeffs
            )

            # Compute the gradients
            enc_grad_input = enc_grad_output.ckks_encrypted_apply_mask(
                enc_backward_output
            )

        return enc_grad_input, None, None
