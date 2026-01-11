from typing import Tuple, Optional
from torchseal.wrapper import CKKSWrapper, CKKSPoolingFunctionWrapper

import typing
import torch


class AvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSPoolingFunctionWrapper, enc_input: CKKSWrapper, weight: torch.Tensor, conv2d_padding_transformation: torch.Tensor, conv2d_inverse_padding_transformation: torch.Tensor) -> CKKSWrapper:
        # Apply the padding transformation to the encrypted input
        # NOTE: This is useless if padding is 0 (we can skip this step if that's the case)
        enc_padding_input = enc_input
        # .ckks_matrix_multiplication(
        #     conv2d_padding_transformation
        # )

        # Save the ctx for the backward method
        ctx.save_for_backward(weight, conv2d_inverse_padding_transformation)

        # Apply the convolution to the encrypted input
        enc_output = enc_padding_input.ckks_matrix_multiplication(weight.t())

        return enc_output

    @staticmethod
    def backward(ctx: CKKSPoolingFunctionWrapper, enc_grad_output: CKKSWrapper) -> Tuple[Optional[CKKSWrapper], None, None, None]:
        # Get the saved tensors
        weight, conv2d_inverse_padding_transformation = typing.cast(
            Tuple[torch.Tensor, torch.Tensor],
            ctx.saved_tensors
        )

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[bool, bool, bool, bool],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        enc_grad_input = None

        # Calculate the gradient for the input
        if result[0]:
            # Calculate the gradients for the input tensor
            enc_padded_grad_input = enc_grad_output.ckks_matrix_multiplication(
                weight
            )

            # Apply the inverse padding transformation to the gradient input
            # NOTE: This is useless if padding is 0 (we can skip this step if that's the case)
            enc_grad_input = enc_padded_grad_input
            # .ckks_matrix_multiplication(
            #     conv2d_inverse_padding_transformation
            # )

        return enc_grad_input, None, None, None
