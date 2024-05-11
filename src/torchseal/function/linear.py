from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSLinearFunctionWrapper

import typing
import torch


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSLinearFunctionWrapper, enc_input: CKKSWrapper, weight: torch.Tensor, bias: torch.Tensor) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight)
        ctx.enc_input = enc_input.clone()

        # Apply the linear transformation to the encrypted input
        enc_output = enc_input.do_matrix_multiplication(
            weight.t()
        ).do_addition(bias)

        return enc_output

    # TODO: Move this to train mode
    @staticmethod
    def backward(ctx: CKKSLinearFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Get the saved tensors
        saved_tensors = typing.cast(
            Tuple[torch.Tensor],
            ctx.saved_tensors
        )
        input = ctx.enc_input.do_decryption()

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
            grad_weight = grad_output.t().matmul(input)
        if result[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias
