from typing import Tuple, Optional

from tenseal import plain_tensor
from torchseal.wrapper import CKKSWrapper, CKKSLinearFunctionWrapper

import typing
import torch
import torchseal


class Conv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSLinearFunctionWrapper, enc_input: CKKSWrapper, weight: CKKSWrapper, bias: CKKSWrapper, conv2d_padding_transformation: torch.Tensor, conv2d_inverse_padding_transformation: torch.Tensor, conv2d_weight_mask: torch.Tensor, conv2d_bias_transformation: torch.Tensor, training: bool) -> CKKSWrapper:
        # Apply the padding transformation to the encrypted input
        # NOTE: This is useless if padding is 0 (we can skip this step if that's the case)
        enc_padding_input = enc_input.ckks_matrix_multiplication(
            conv2d_padding_transformation
        )

        # Save the ctx for the backward method
        ctx.save_for_backward(
            conv2d_inverse_padding_transformation,
            conv2d_weight_mask,
            conv2d_bias_transformation
        )
        ctx.enc_input = enc_padding_input.clone()
        ctx.weight = weight.clone()

        # Apply the convolution to the encrypted input
        # If training, apply the linear transformation to the encrypted input using encrypted parameters
        if training:
            enc_output = enc_padding_input.ckks_encrypted_matrix_multiplication(
                weight.ckks_transpose()
            ).ckks_encrypted_addition(bias)

            return enc_output

        # Else, apply the linear transformation to the encrypted input using plaintext parameters
        enc_output = enc_padding_input.ckks_matrix_multiplication(
            weight.plaintext_data.t()
        ).ckks_addition(bias.plaintext_data)

        return enc_output

    @staticmethod
    def backward(ctx: CKKSLinearFunctionWrapper, enc_grad_output: CKKSWrapper) -> Tuple[Optional[CKKSWrapper], Optional[CKKSWrapper], Optional[CKKSWrapper], None, None, None, None, None]:
        # Get the saved tensors
        conv2d_inverse_padding_transformation, conv2d_weight_mask, conv2d_bias_transformation = typing.cast(
            Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor
            ],
            ctx.saved_tensors
        )
        enc_input = ctx.enc_input
        enc_weight = ctx.weight

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[bool, bool, bool, bool, bool, bool, bool, bool],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        enc_grad_input = enc_grad_weight = enc_grad_bias = None

        if result[0]:
            # Calculate the gradients for the input tensor (this will be encrypted)
            enc_padded_grad_input = enc_grad_output.ckks_encrypted_matrix_multiplication(
                enc_weight
            )

            # Apply the inverse padding transformation to the gradient input
            # NOTE: This is useless if padding is 0 (we can skip this step if that's the case)
            enc_grad_input = enc_padded_grad_input.ckks_matrix_multiplication(
                conv2d_inverse_padding_transformation
            )

        if result[1]:
            # Create the fully connected gradient weight tensor (this will be encrypted)
            enc_unfiltered_grad_weight = enc_grad_output.ckks_transpose().ckks_encrypted_matrix_multiplication(
                enc_input
            )

            # Initialize the gradient weight tensor (this will be encrypted, probably going to be plain tensor)
            enc_grad_weight = torchseal.ckks_zeros(
                enc_weight.shape, do_encryption=True
            )

            # Apply the binary tensor to the gradient weight
            for binary_mask in conv2d_weight_mask:
                # Apply the binary mask to the gradient weight (this will be encrypted)
                enc_filtered_grad_weight = enc_unfiltered_grad_weight.ckks_apply_mask(
                    binary_mask
                )

                # Calculate the sum of all elements (this will be encrypted)
                enc_sum_all_element = enc_filtered_grad_weight.ckks_sum(
                    axis=1
                ).ckks_sum(
                    axis=0
                )

                # Create the new current gradient weight (this will be encrypted)
                enc_current_gradient_weight = enc_sum_all_element.ckks_apply_mask(
                    binary_mask
                )

                # Add the current gradient weight to the final gradient weight (this will be encrypted)
                enc_grad_weight = enc_grad_weight.ckks_encrypted_addition(
                    enc_current_gradient_weight
                )

        if result[2]:
            # Apply the binary transformation to the gradient output (this will be encrypted)
            enc_linear_grad_bias = enc_grad_output.ckks_sum(
                axis=0
            )

            # Calculate the gradient bias for the convolutional layer (this will be encrypted)
            enc_grad_bias = enc_linear_grad_bias.ckks_apply_transformation(
                conv2d_bias_transformation
            )

        return enc_grad_input, enc_grad_weight, enc_grad_bias, None, None, None, None, None
