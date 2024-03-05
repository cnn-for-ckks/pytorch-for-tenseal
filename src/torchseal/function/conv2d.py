from typing import Tuple, Optional
from torch import Tensor
from torch.autograd import Function
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.wrapper.function import CKKSFunctionWrapper


class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx: CKKSFunctionWrapper, enc_x: CKKSWrapper, weight: Tensor, bias: Tensor, windows_nb: int) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight)
        ctx.enc_x = enc_x.clone()

        # Apply the convolution to the encrypted input
        out_x = enc_x.do_conv2d(weight, bias, windows_nb)

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
        grad_input = grad_weight = grad_bias = None  # TODO: Compute the gradients

        return grad_input, grad_weight, grad_bias
