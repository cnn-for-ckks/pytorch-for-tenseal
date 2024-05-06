from typing import Tuple, Optional
from torchseal.wrapper import CKKSWrapper, CKKSPoolingFunctionWrapper

import typing
import torch


class AvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSPoolingFunctionWrapper, enc_x: CKKSWrapper, weight: torch.Tensor, conv2d_input_mask: torch.Tensor) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight, conv2d_input_mask)

        # Apply the convolution to the encrypted input
        out_x = enc_x.do_linear(weight)

        return out_x

    @staticmethod
    def backward(ctx: CKKSPoolingFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None]:
        # Get the saved tensors
        weight, conv2d_input_mask = typing.cast(
            Tuple[torch.Tensor, torch.Tensor],
            ctx.saved_tensors
        )

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[bool, bool, bool],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        grad_input = None

        # Calculate the gradient for the input
        if result[0]:
            # Calculate the gradients for the input tensor (this will be encrypted)
            grad_input = grad_output.matmul(weight).mul(conv2d_input_mask)

        return grad_input, None, None
