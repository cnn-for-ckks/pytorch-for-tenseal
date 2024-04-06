from typing import Callable, Optional, Tuple
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.wrapper.function import CKKSActivationFunctionWrapper

import torch
import numpy as np


class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSActivationFunctionWrapper, enc_x: CKKSWrapper, coeffs: np.ndarray, approx_poly_derivative: Callable[[float], float]) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.enc_x = enc_x.clone()
        ctx.polyval_derivative = approx_poly_derivative

        # Apply square function to the encrypted input
        out_x = enc_x.do_activation_function(coeffs)

        return out_x

    @staticmethod
    def backward(ctx: CKKSActivationFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Get the saved tensors
        x = ctx.enc_x.do_decryption()
        polyval_derivative = ctx.polyval_derivative

        # Do the backward operation
        out = x.do_activation_function_backward(polyval_derivative)

        # Get the needs_input_grad
        result: Tuple[bool, bool, bool] = ctx.needs_input_grad  # type: ignore

        # Initialize the gradients
        grad_input = None

        if result[0]:
            # Compute the gradients
            grad_input = grad_output.mul(out)

        return grad_input, None, None
