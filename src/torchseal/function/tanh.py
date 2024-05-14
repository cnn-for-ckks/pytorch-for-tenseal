from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSActivationFunctionWrapper

import typing
import numpy as np
import torch
import torchseal


class TanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSActivationFunctionWrapper, enc_input: CKKSWrapper, coeffs: np.ndarray) -> CKKSWrapper:
        # Apply the sigmoid function to the encrypted input
        enc_output = enc_input.ckks_encrypted_polynomial(coeffs)

        # Save the ctx for the backward method
        ctx.enc_output = enc_output.clone()

        return enc_output

    @staticmethod
    def backward(ctx: CKKSActivationFunctionWrapper, enc_grad_output: CKKSWrapper) -> Tuple[Optional[CKKSWrapper], None]:
        # Get the saved tensors
        enc_output = ctx.enc_output

        # Get the needs_input_grad
        result = typing.cast(Tuple[bool, bool], ctx.needs_input_grad)

        # Initialize the gradients
        enc_grad_input = None

        if result[0]:
            # Compute the gradients (grad_output * (1 - output.pow(2)))
            enc_grad_input = enc_grad_output.ckks_encrypted_apply_mask(
                torchseal.ckks_ones(enc_output.shape, do_encryption=True).ckks_encrypted_addition(
                    enc_output.ckks_encrypted_square().ckks_encrypted_negation()
                )
            )

        return enc_grad_input, None
