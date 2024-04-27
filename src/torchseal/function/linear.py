from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSFunctionWrapper

import typing
import torch


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSFunctionWrapper, enc_x: CKKSWrapper, weight: torch.Tensor, bias: torch.Tensor) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight)
        ctx.enc_x = enc_x.clone()

        # Apply the linear transformation to the encrypted input
        out_x = enc_x.do_linear(weight, bias)

        return out_x

    @staticmethod
    def backward(ctx: CKKSFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Get the saved tensors
        saved_tensors = typing.cast(
            Tuple[torch.Tensor],
            ctx.saved_tensors
        )
        x = ctx.enc_x.do_decryption()

        # Unpack the saved tensors
        weight, = saved_tensors

        # Get the needs_input_grad
        result = typing.cast(Tuple[bool, bool, bool], ctx.needs_input_grad)

        # Initialize the gradients
        grad_input = grad_weight = grad_bias = None

        # Compute the gradients
        if result[0]:
            grad_input = grad_output.matmul(weight)
        if result[1]:
            grad_weight = grad_output.t().matmul(x)
        if result[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
