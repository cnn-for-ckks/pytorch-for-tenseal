from typing import Tuple, Optional
from torch.nn.grad import conv2d_input, conv2d_weight
from torchseal.wrapper import CKKSWrapper, CKKSConvFunctionWrapper
from torchseal.utils import approximate_toeplitz_multiple_channels

import typing
import torch


class Conv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSConvFunctionWrapper, enc_x: CKKSWrapper, weight: torch.Tensor, bias: torch.Tensor, input_size_with_channel: Tuple[int, int, int, int], stride: int, padding: int) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight)
        ctx.enc_x = enc_x.clone()
        ctx.input_size_with_channel = input_size_with_channel
        ctx.stride = stride
        ctx.padding = padding

        # Get the toeplitz weight
        toeplitz_weight = approximate_toeplitz_multiple_channels(
            weight, input_size_with_channel[1:], stride=stride, padding=padding
        )

        # Get the bias
        toeplitz_output_length, _ = toeplitz_weight.shape
        bias_length, = bias.shape
        toeplitz_bias = bias.repeat_interleave(
            toeplitz_output_length // bias_length
        )

        # Apply the convolution to the encrypted input
        out_x = enc_x.do_linear(toeplitz_weight, toeplitz_bias)

        return out_x

    @staticmethod
    def backward(ctx: CKKSConvFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], None, None, None]:
        # Get the saved tensors
        saved_tensors = typing.cast(Tuple[torch.Tensor], ctx.saved_tensors)
        x = ctx.enc_x.do_decryption()
        input_size_with_channel = ctx.input_size_with_channel
        stride = ctx.stride
        padding = ctx.padding

        # Unpack the saved tensors
        weight, = saved_tensors

        # Unpack the tensor shapes
        batch_size, input_channel, input_height, input_width = input_size_with_channel
        kernel_out_channel, _, kernel_height, kernel_width = weight.shape

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
        grad_input = grad_weight = grad_bias = None

        if result[0]:
            grad_input = conv2d_input(
                reshaped_x.shape, weight, reshaped_grad_output, stride=stride, padding=padding
            ).view(batch_size, -1)
        if result[1]:
            grad_weight = conv2d_weight(
                reshaped_x, weight.shape, reshaped_grad_output, stride=stride, padding=padding
            )
        if result[2]:
            grad_bias = reshaped_grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None
