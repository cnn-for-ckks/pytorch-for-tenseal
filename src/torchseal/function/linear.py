from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSLinearFunctionWrapper

import typing
import torch


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSLinearFunctionWrapper, enc_input: CKKSWrapper, weight: CKKSWrapper, bias: CKKSWrapper, training: bool) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.enc_input = enc_input.clone()
        ctx.weight = weight.clone()

        # Apply the linear transformation to the encrypted input
        # If training, apply the linear transformation to the encrypted input using encrypted parameters
        if training:
            enc_output = enc_input.ckks_encrypted_matrix_multiplication(
                weight.ckks_transpose()
            ).ckks_encrypted_addition(bias)

            return enc_output

        # Else, apply the linear transformation to the encrypted input using plaintext parameters
        enc_output = enc_input.ckks_matrix_multiplication(
            weight.plaintext_data.t()
        ).ckks_addition(bias.plaintext_data)

        return enc_output

    @staticmethod
    def backward(ctx: CKKSLinearFunctionWrapper, enc_grad_output: CKKSWrapper) -> Tuple[Optional[CKKSWrapper], Optional[CKKSWrapper], Optional[CKKSWrapper], None]:
        # Get the saved tensors
        enc_weight = ctx.weight
        enc_input = ctx.enc_input

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[bool, bool, bool, bool],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        enc_grad_input = enc_grad_weight = enc_grad_bias = None

        # Compute the gradients
        if result[0]:
            enc_grad_input = enc_grad_output.ckks_encrypted_matrix_multiplication(
                enc_weight
            )
        if result[1]:
            enc_grad_weight = enc_grad_output.ckks_transpose().ckks_encrypted_matrix_multiplication(
                enc_input
            )
        if result[2]:
            enc_grad_bias = enc_grad_output.ckks_sum(axis=0)

        return enc_grad_input, enc_grad_weight, enc_grad_bias, None
