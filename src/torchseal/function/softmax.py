from typing import Optional, Tuple
from torchseal.wrapper import CKKSWrapper, CKKSSoftmaxFunctionWrapper

import typing
import numpy as np
import torch


class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CKKSSoftmaxFunctionWrapper, enc_x: CKKSWrapper, exp_coeffs: np.ndarray, inverse_coeffs: np.ndarray, iterations: int) -> CKKSWrapper:
        # Apply the exp function to the encrypted input
        act_x = enc_x.do_activation_function(exp_coeffs)
        act_x_copy = act_x.clone()

        # Apply the sum function to the encrypted input
        sum_x = act_x.do_sum(axis=1)

        # Apply the multiplicative inverse function to the encrypted input
        inverse_sum_x = sum_x.do_multiplicative_inverse(
            inverse_coeffs, iterations
        )

        # Apply the division function to the encrypted input
        out_x = act_x_copy.do_element_multiplication(inverse_sum_x)

        # Save the ctx for the backward method
        ctx.out_x = out_x.clone()

        return out_x

    @staticmethod
    def backward(ctx: CKKSSoftmaxFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None, None, None]:
        # Get the saved tensors
        out_x = ctx.out_x.do_decryption()

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[bool, bool, bool, bool],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        grad_input = None

        if result[0]:
            # Compute the gradients
            grad_input = out_x * \
                (grad_output - (grad_output * out_x).sum(dim=1, keepdim=True))

        return grad_input, None, None, None
