from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSLossFunctionWrapper

import typing
import numpy as np
import torch


class CrossEntropyLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSLossFunctionWrapper, enc_input: CKKSWrapper, enc_target: CKKSWrapper, exp_coeffs: np.ndarray, inverse_coeffs: np.ndarray, inverse_iterations: int, log_coeffs: np.ndarray) -> CKKSWrapper:
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

    # TODO: Move this to encrypted mode
    @staticmethod
    def backward(ctx: CKKSLossFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None, None, None, None]:
        # Get the saved tensors
        output = ctx.enc_output.decrypt()
        target = ctx.enc_target.decrypt()

        # Unpack the target shape
        batch_size, _ = target.shape

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[
                bool, bool, bool, bool, bool, bool
            ],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        grad_input = None

        if result[0]:
            # Add the subtraction mask to the output (will be approximated for encrypted data)
            output_subtracted = output.subtract(target)

            # Multiply by the grad_output (will be approximated for encrypted data)
            grad_input = grad_output.mul(output_subtracted).mul(1 / batch_size)

        return grad_input, None, None, None, None, None
