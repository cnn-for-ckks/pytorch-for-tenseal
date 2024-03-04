from typing import Tuple, Optional
from torch import Tensor
from torch.autograd import Function
from tenseal import CKKSVector
from torchseal.wrapper.function import CKKSFunctionCtx

import torch


class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx: CKKSFunctionCtx, enc_x: CKKSVector, weight: Tensor, bias: Tensor, windows_nb: int) -> CKKSVector:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight)
        ctx.enc_x = enc_x

        # TODO: Move pack_vectors to the "Flatten" layer
        return CKKSVector.pack_vectors([
            enc_x.conv2d_im2col(kernel, windows_nb).add(bias) for kernel, bias in zip(weight.tolist(), bias.tolist())
        ])

    # NOTE: This method requires private key access
    @staticmethod
    def backward(ctx: CKKSFunctionCtx, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        # Get the saved tensors
        saved_tensors: Tuple[Tensor] = ctx.saved_tensors  # type: ignore
        enc_x = ctx.enc_x

        # Unpack the saved tensors
        weight, = saved_tensors

        # Get the needs_input_grad
        result: Tuple[bool, bool, bool] = ctx.needs_input_grad  # type: ignore

        # Decrypt the encrypted input and gradient
        x = torch.tensor(enc_x.decrypt(), requires_grad=True)

        # Initialize the gradients
        grad_input = grad_weight = grad_bias = None

        # TODO: Compute the gradients

        return grad_input, grad_weight, grad_bias
