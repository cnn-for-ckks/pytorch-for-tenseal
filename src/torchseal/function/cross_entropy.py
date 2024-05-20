from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSLossFunctionWithOutputWrapper

import typing
import numpy as np
import torch


class CrossEntropyLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSLossFunctionWithOutputWrapper, enc_input: CKKSWrapper, enc_target: CKKSWrapper, exp_coeffs: np.ndarray, inverse_coeffs: np.ndarray, inverse_iterations: int, log_coeffs: np.ndarray) -> CKKSWrapper:
        # Apply softmax to the input to get probabilities
        enc_output = enc_input.ckks_encrypted_softmax(
            exp_coeffs, inverse_coeffs, inverse_iterations
        )

        # Save input and target for backward pass
        ctx.enc_output = enc_output.clone()
        ctx.enc_target = enc_target.clone()

        # Get the batch size
        batch_size, _ = enc_input.shape

        # Calculate the loss
        enc_loss = enc_output.ckks_encrypted_negative_log_likelihood_loss(
            enc_target, log_coeffs, batch_size
        )

        return enc_loss

    @staticmethod
    def backward(ctx: CKKSLossFunctionWithOutputWrapper, enc_grad_output: CKKSWrapper) -> Tuple[Optional[CKKSWrapper], None, None, None, None, None]:
        # Get the saved tensors
        enc_output = ctx.enc_output
        enc_target = ctx.enc_target

        # Unpack the target shape
        batch_size, _ = enc_target.shape

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[
                bool, bool, bool, bool, bool, bool
            ],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        enc_grad_input = None

        if result[0]:
            # Add the subtraction mask to the output (this will be encrypted)
            enc_output_subtracted = enc_output.ckks_encrypted_addition(
                enc_target.ckks_encrypted_negation()
            )

            # Multiply by the grad_output (this will be encrypted)
            enc_grad_input = enc_output_subtracted.ckks_encrypted_apply_mask(
                enc_grad_output
            ).ckks_apply_scalar(1 / batch_size)

        return enc_grad_input, None, None, None, None, None
