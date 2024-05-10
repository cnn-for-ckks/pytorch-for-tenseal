from typing import Optional, Tuple
from numpy.polynomial import Polynomial
from torchseal.wrapper import CKKSWrapper, CKKSActivationFunctionWrapper

import typing
import numpy as np
import torch


class TanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSActivationFunctionWrapper, enc_input: CKKSWrapper, coeffs: np.ndarray, deriv_coeffs: np.ndarray) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.enc_input = enc_input.clone()
        ctx.deriv_coeffs = deriv_coeffs

        # Apply the sigmoid function to the encrypted input
        enc_output = enc_input.do_polynomial(coeffs)

        return enc_output

    @staticmethod
    def backward(ctx: CKKSActivationFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None]:
        # Get the saved tensors
        input = ctx.enc_input.do_decryption()
        deriv_coeffs = ctx.deriv_coeffs

        # Do the backward operation
        polynomial_deriv = Polynomial(deriv_coeffs)
        backward_output = torch.tensor(
            np.vectorize(polynomial_deriv)(input)
        )

        # Get the needs_input_grad
        result = typing.cast(Tuple[bool, bool, bool], ctx.needs_input_grad)

        # Initialize the gradients
        grad_input = None

        if result[0]:
            # Compute the gradients
            grad_input = grad_output.mul(backward_output)

        return grad_input, None, None
