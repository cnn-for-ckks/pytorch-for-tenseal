from typing import Tuple, Optional
from torchseal.wrapper import CKKSWrapper, CKKSFunctionWrapper

import typing
import torch


class Conv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSFunctionWrapper, enc_x: CKKSWrapper, weight: torch.Tensor, bias: torch.Tensor, conv2d_input_mask: torch.Tensor, conv2d_weight_mask: torch.Tensor, conv2d_bias_transformation: torch.Tensor) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.save_for_backward(
            weight,
            conv2d_input_mask,
            conv2d_weight_mask,
            conv2d_bias_transformation
        )
        ctx.enc_x = enc_x.clone()

        # Apply the convolution to the encrypted input
        out_x = enc_x.do_linear(weight, bias)

        return out_x

    @staticmethod
    def backward(ctx: CKKSFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], None, None, None]:
        # Get the saved tensors
        weight, conv2d_input_mask, conv2d_weight_mask, conv2d_bias_transformation = typing.cast(
            Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ],
            ctx.saved_tensors
        )
        x = ctx.enc_x.do_decryption()

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[bool, bool, bool, bool, bool, bool],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        grad_input = grad_weight = grad_bias = None

        if result[0]:
            # Calculate the gradients for the input tensor (this will be encrypted)
            grad_input = grad_output.matmul(weight).mul(conv2d_input_mask)

        if result[1]:
            # Create the fully connected gradient weight tensor (this will be encrypted)
            unfiltered_grad_weight = grad_output.t().matmul(x)

            # Initialize the gradient weight tensor (this will be encrypted, probably going to need context with public keys)
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
            grad_bias = conv2d_bias_transformation.matmul(grad_output.sum(0))

        return grad_input, grad_weight, grad_bias, None, None, None
