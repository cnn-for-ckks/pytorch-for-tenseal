from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSLossFunctionWrapper

import typing
import torch


class MSELossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSLossFunctionWrapper, enc_input: CKKSWrapper, enc_target: CKKSWrapper) -> CKKSWrapper:
        # Save input and target for backward pass
        ctx.enc_input = enc_input.clone()
        ctx.enc_target = enc_target.clone()

        # Calculate the difference between the input and the target
        enc_diff = enc_input.ckks_encrypted_addition(
            enc_target.ckks_encrypted_negation()
        )

        # Calculate the square of the difference
        enc_diff_squared = enc_diff.ckks_encrypted_square()

        # Sum the squared differences
        enc_diff_squared_sum = enc_diff_squared.ckks_sum(
            axis=1
        ).ckks_sum(
            axis=0
        )

        # Get the number of elements in the input
        batch_size, input_dim = enc_input.shape
        numel = batch_size * input_dim

        # Calculate the mean of the loss
        enc_loss = enc_diff_squared_sum.ckks_apply_scalar(1 / numel)

        return enc_loss

    @staticmethod
    def backward(ctx: CKKSLossFunctionWrapper, enc_grad_output: CKKSWrapper) -> Tuple[Optional[CKKSWrapper], None]:
        # Get the saved tensors
        enc_input = ctx.enc_input
        enc_target = ctx.enc_target

        # Unpack the input shape
        batch_size, input_dim = enc_input.shape
        numel = batch_size * input_dim

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[
                bool, bool
            ],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        enc_grad_input = None

        if result[0]:
            # Calculate the subtraction mask
            enc_input_subtracted = enc_input.ckks_encrypted_addition(
                enc_target.ckks_encrypted_negation()
            ).ckks_apply_scalar(2 / numel)

            # Multiply by the grad_output
            enc_grad_input = enc_input_subtracted.ckks_encrypted_apply_mask(
                enc_grad_output
            )

        return enc_grad_input, None
