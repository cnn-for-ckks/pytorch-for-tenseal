from typing import Tuple, Optional
from torch import Tensor
from torch.autograd import Function
from torch.nn.grad import conv2d_input, conv2d_weight
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.wrapper.function import CKKSConvFunctionWrapper
from torchseal.utils import toeplitz_multiple_channels

import torch


class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx: CKKSConvFunctionWrapper, enc_x: CKKSWrapper, weight: Tensor, bias: Tensor, output_size: Tuple[int, int], kernel_size: Tuple[int, int]) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight)
        ctx.enc_x = enc_x.clone()
        ctx.output_size = output_size
        ctx.kernel_size = kernel_size

        # Apply the convolution to the encrypted input
        # TODO: Add stride and padding support
        toeplitz_weight = toeplitz_multiple_channels(
            weight, torch.Size([1, *output_size])
        )
        out_x = enc_x.do_conv2d(toeplitz_weight, bias)

        return out_x

    @staticmethod
    def backward(ctx: CKKSConvFunctionWrapper, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        # Get the saved tensors
        saved_tensors: Tuple[Tensor] = ctx.saved_tensors  # type: ignore
        enc_x = ctx.enc_x
        output_size = ctx.output_size
        kernel_size = ctx.kernel_size

        # TODO: Add stride and padding support
        stride = 1
        padding = 0

        # Unpack the saved tensors
        weight, = saved_tensors

        # Get the needs_input_grad
        result: Tuple[bool, bool, bool] = ctx.needs_input_grad  # type: ignore

        # Calculate feature dimension
        feature_h = (
            output_size[0] - kernel_size[0] + 2 * padding
        ) // stride + 1
        feature_w = (
            output_size[1] - kernel_size[1] + 2 * padding
        ) // stride + 1

        # Decrypt the input
        reshaped_x = enc_x.do_decryption().view(
            1, 1, output_size[0], output_size[1]
        )
        reshaped_grad_output = grad_output.view(
            1, 1, feature_h, feature_w
        )

        # Initialize the gradients
        grad_input = grad_weight = grad_bias = None

        if result[0]:
            grad_input = conv2d_input(
                reshaped_x.shape, weight, reshaped_grad_output, stride=stride
            ).view(-1)
        if result[1]:
            grad_weight = conv2d_weight(
                reshaped_x, weight.shape, reshaped_grad_output, stride=stride
            ).view(weight.shape)
        if result[2]:
            grad_bias = grad_output

        return grad_input, grad_weight, grad_bias, None, None
