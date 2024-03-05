from typing import Optional, Tuple
from torch import Tensor
from torch.autograd import Function
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.wrapper.function import CKKSFunctionWrapper


class LinearFunction(Function):
    @staticmethod
    def forward(ctx: CKKSFunctionWrapper, enc_x: CKKSWrapper, weight: Tensor, bias: Tensor) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight)
        ctx.enc_x = enc_x.clone()

        # Apply the linear transformation to the encrypted input
        out_x = enc_x.do_linear(weight, bias)

        return out_x

    # NOTE: This method requires private key access
    @staticmethod
    def backward(ctx: CKKSFunctionWrapper, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        # Get the saved tensors
        saved_tensors: Tuple[Tensor] = ctx.saved_tensors  # type: ignore
        x = ctx.enc_x.do_decryption()

        # Unpack the saved tensors
        weight, = saved_tensors

        # Get the needs_input_grad
        result: Tuple[bool, bool, bool] = ctx.needs_input_grad  # type: ignore

        # Initialize the gradients
        grad_input = grad_weight = grad_bias = None

        # Compute the gradients
        if result[0]:
            grad_input = grad_output.matmul(weight)
        if result[1]:
            grad_weight = grad_output.unsqueeze(0).t().matmul(x.unsqueeze(0))
        if result[2]:
            grad_bias = grad_output

        return grad_input, grad_weight, grad_bias

    @staticmethod
    def apply(enc_x: CKKSWrapper, weight: Tensor, bias: Tensor) -> CKKSWrapper:
        out_x: CKKSWrapper = super(LinearFunction, LinearFunction).apply(
            enc_x, weight, bias
        )  # type: ignore

        return out_x
