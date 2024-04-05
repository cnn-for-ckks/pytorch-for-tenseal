from typing import Tuple, Optional
from torch.nn.grad import conv2d_input
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.wrapper.function import CKKSConvFunctionWrapper

import torch


class AvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSConvFunctionWrapper, enc_x: CKKSWrapper, avg_kernel: torch.Tensor, toeplitz_avg_kernel: torch.Tensor, output_size: torch.Size, stride: int, padding: int) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.save_for_backward(avg_kernel)
        ctx.enc_x = enc_x.clone()
        ctx.output_size = output_size
        ctx.stride = stride
        ctx.padding = padding

        # Apply the convolution to the encrypted input
        out_x = enc_x.do_multiplication(toeplitz_avg_kernel)

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
        avg_kernel, = saved_tensors

        # Unpack the tensor shapes
        output_channel, output_height, output_width = output_size
        kernel_out_channel, _, kernel_height, kernel_width = avg_kernel.shape

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
            1, kernel_out_channel, feature_h, feature_w  # TODO: Handle larger batch sizes
        )

        # Get the needs_input_grad
        result: Tuple[bool] = ctx.needs_input_grad  # type: ignore

        # Initialize the gradients
        grad_input = None

        # Calculate the gradient for the input
        if result[0]:
            grad_input = conv2d_input(
                reshaped_x.shape, avg_kernel, reshaped_grad_output, stride=stride, padding=padding
            ).view(-1)

        return grad_input, None, None, None, None, None
