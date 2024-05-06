from typing import Tuple, Optional
from torch.nn.grad import conv2d_input
from torchseal.wrapper import CKKSWrapper, CKKSConvFunctionWrapper

import typing
import torch


class AvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSConvFunctionWrapper, enc_x: CKKSWrapper, avg_kernel: torch.Tensor, toeplitz_avg_kernel: torch.Tensor, input_size_with_channel: Tuple[int, int, int, int], stride: int, padding: int) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.save_for_backward(avg_kernel)
        ctx.enc_x = enc_x.clone()
        ctx.input_size_with_channel = input_size_with_channel
        ctx.stride = stride
        ctx.padding = padding

        # Apply the convolution to the encrypted input
        out_x = enc_x.do_linear(toeplitz_avg_kernel)

        return out_x

    @staticmethod
    def backward(ctx: CKKSConvFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None, None, None, None]:
        # Get the saved tensors
        saved_tensors = typing.cast(Tuple[torch.Tensor], ctx.saved_tensors)
        x = ctx.enc_x.do_decryption()
        input_size_with_channel = ctx.input_size_with_channel
        stride = ctx.stride
        padding = ctx.padding

        # Unpack the saved tensors
        avg_kernel, = saved_tensors

        # Unpack the tensor shapes
        batch_size, input_channel, input_height, input_width = input_size_with_channel
        kernel_out_channel, _, kernel_height, kernel_width = avg_kernel.shape

        # Calculate feature dimension
        feature_h = (
            input_height - kernel_height + 2 * padding
        ) // stride + 1
        feature_w = (
            input_width - kernel_width + 2 * padding
        ) // stride + 1

        # Decrypt the input
        reshaped_x = x.view(
            batch_size, input_channel, input_height, input_width
        )
        reshaped_grad_output = grad_output.view(
            batch_size, kernel_out_channel, feature_h, feature_w
        )

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[bool, bool, bool, bool, bool, bool],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        grad_input = None

        # Calculate the gradient for the input
        if result[0]:
            grad_input = conv2d_input(
                reshaped_x.shape, avg_kernel, reshaped_grad_output, stride=stride, padding=padding
            ).view(batch_size, -1)

        return grad_input, None, None, None, None, None
