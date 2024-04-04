from typing import Tuple, Optional
from torch.nn.grad import conv2d_input, conv2d_weight
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.wrapper.function import CKKSConvFunctionWrapper
from torchseal.utils import toeplitz_multiple_channels

import torch


class Conv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSConvFunctionWrapper, enc_x: CKKSWrapper, weight: torch.Tensor, bias: torch.Tensor, output_size: torch.Size, stride: int, padding: int) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight)
        ctx.enc_x = enc_x.clone()
        ctx.output_size = output_size
        ctx.stride = stride
        ctx.padding = padding

        # Get the toeplitz weight
        toeplitz_weight = toeplitz_multiple_channels(
            weight, output_size, stride=stride, padding=padding
        )

        # Get the bias
        toeplitz_output_length, _ = toeplitz_weight.shape
        bias_length, = bias.shape
        toeplitz_bias = torch.cat(
            [bias for _ in range(toeplitz_output_length // bias_length)]
        )

        # Apply the convolution to the encrypted input
        out_x = enc_x.do_linear(toeplitz_weight, toeplitz_bias)

        return out_x

    @staticmethod
    def backward(ctx: CKKSConvFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Get the saved tensors
        saved_tensors: Tuple[torch.Tensor] = ctx.saved_tensors  # type: ignore
        enc_x = ctx.enc_x
        output_size = ctx.output_size
        stride = ctx.stride
        padding = ctx.padding

        # Unpack the saved tensors
        weight, = saved_tensors

        # Unpack the tensor shapes
        output_channel, output_height, output_width = output_size
        kernel_out_channel, kernel_in_channel, kernel_height, kernel_width = weight.shape

        # Calculate feature dimension
        feature_h = (
            output_height - kernel_height + 2 * padding
        ) // stride + 1
        feature_w = (
            output_width - kernel_width + 2 * padding
        ) // stride + 1

        # Decrypt the input
        reshaped_x = enc_x.do_decryption().view(
            1, output_channel, output_height, output_width  # TODO: Handle larger batch sizes
        )
        reshaped_grad_output = grad_output.view(
            kernel_out_channel, kernel_in_channel, feature_h, feature_w
        )

        # Get the needs_input_grad
        result: Tuple[bool, bool, bool] = ctx.needs_input_grad  # type: ignore

        # Initialize the gradients
        grad_input = grad_weight = grad_bias = None

        if result[0]:
            grad_input = conv2d_input(
                reshaped_x.shape, weight, reshaped_grad_output, stride=stride, padding=padding
            ).view(-1)
        if result[1]:
            grad_weight = conv2d_weight(
                reshaped_x, weight.shape, reshaped_grad_output, stride=stride, padding=padding
            )
        if result[2]:
            grad_bias = grad_output

        return grad_input, grad_weight, grad_bias, None, None, None
