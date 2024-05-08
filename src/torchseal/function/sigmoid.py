from typing import Callable, Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSActivationFunctionWrapper

import typing
import numpy as np
import torch


class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSActivationFunctionWrapper, enc_input: CKKSWrapper, coeffs: np.ndarray, approx_poly_derivative: Callable[[float], float]) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.enc_input = enc_input.clone()
        ctx.polyval_derivative = approx_poly_derivative

        # Apply the sigmoid function to the encrypted input
        enc_output = enc_input.do_activation_function(coeffs)

        return enc_output

    @staticmethod
    def backward(ctx: CKKSActivationFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None]:
        # Get the saved tensors
        input = ctx.enc_input.do_decryption()
        polyval_derivative = ctx.polyval_derivative

        # Do the backward operation
        backward_output = input.do_activation_function_backward(
            polyval_derivative
        )

        # Get the needs_input_grad
        result = typing.cast(Tuple[bool, bool, bool], ctx.needs_input_grad)

        # Initialize the gradients
        grad_input = None

        if result[0]:
            # Compute the gradients
            grad_input = grad_output.mul(backward_output)

        return grad_input, None, None
