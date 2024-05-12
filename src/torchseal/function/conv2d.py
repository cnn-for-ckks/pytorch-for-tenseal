from typing import Tuple, Optional
from torchseal.wrapper import CKKSWrapper, CKKSLinearFunctionWrapper

import typing
import torch


class Conv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSLinearFunctionWrapper, enc_input: CKKSWrapper, weight: torch.Tensor, bias: torch.Tensor, conv2d_padding_transformation: torch.Tensor, conv2d_inverse_padding_transformation: torch.Tensor, conv2d_weight_mask: torch.Tensor, conv2d_bias_transformation: torch.Tensor, training: bool) -> CKKSWrapper:
        # Apply the padding transformation to the encrypted input
        # NOTE: This is useless if padding is 0 (we can skip this step if that's the case)
        enc_padding_input = enc_input.do_matrix_multiplication(
            conv2d_padding_transformation
        )

        # Save the ctx for the backward method
        ctx.save_for_backward(
            weight,
            conv2d_inverse_padding_transformation,
            conv2d_weight_mask,
            conv2d_bias_transformation
        )
        ctx.enc_input = enc_padding_input.clone()
        ctx.training = training

        # Apply the convolution to the encrypted input
        # TODO: Implement the convolution for encrypted parameters
        enc_output = enc_padding_input.do_matrix_multiplication(
            weight.t()
        ).do_addition(bias)

        return enc_output

    # TODO: Move this to encrypted mode
    @staticmethod
    def backward(ctx: CKKSLinearFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], None, None, None, None, None]:
        # Get the saved tensors
        weight, conv2d_inverse_padding_transformation, conv2d_weight_mask, conv2d_bias_transformation = typing.cast(
            Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ],
            ctx.saved_tensors
        )
        input = ctx.enc_input.do_decryption()

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[bool, bool, bool, bool, bool, bool, bool, bool],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        grad_input = grad_weight = grad_bias = None

        if result[0]:
            # Calculate the gradients for the input tensor (this will be encrypted)
            padded_grad_input = grad_output.matmul(weight)

            # Apply the inverse padding transformation to the gradient input
            # NOTE: This is useless if padding is 0 (we can skip this step if that's the case)
            grad_input = padded_grad_input.matmul(
                conv2d_inverse_padding_transformation
            )

        if result[1]:
            # Create the fully connected gradient weight tensor (this will be encrypted)
            unfiltered_grad_weight = grad_output.t().matmul(input)

            # Initialize the gradient weight tensor (this will be encrypted, probably going to be plain tensor)
            grad_weight = torch.zeros_like(weight)

            # Apply the binary tensor to the gradient weight
            for binary_mask in conv2d_weight_mask:
                # Apply the binary mask to the gradient weight (this will be encrypted)
                filtered_grad_weight = unfiltered_grad_weight.mul(binary_mask)

                # Calculate the sum of all elements (this will be encrypted)
                sum_all_element = filtered_grad_weight.sum(1).sum(0)

                # Create the new current gradient weight (this will be encrypted)
                current_gradient_weight = binary_mask.mul(sum_all_element)

                # Add the current gradient weight to the final gradient weight (this will be encrypted)
                grad_weight += current_gradient_weight

        if result[2]:
            # Apply the binary transformation to the gradient output (this will be encrypted)
            linear_grad_bias = grad_output.sum(0)

            # Calculate the gradient bias for the convolutional layer (this will be encrypted)
            grad_bias = conv2d_bias_transformation.matmul(linear_grad_bias)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None
