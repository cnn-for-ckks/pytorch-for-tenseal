from typing import Tuple, Optional
from torch import Tensor
from torch.autograd import Function
from torch.nn.grad import conv2d_input, conv2d_weight
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.wrapper.function import CKKSConvFunctionWrapper


class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx: CKKSConvFunctionWrapper, enc_x: CKKSWrapper, weight: Tensor, bias: Tensor, windows_nb: int, stride: int) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight)
        ctx.enc_x = enc_x.clone()
        ctx.stride = stride

        # Apply the convolution to the encrypted input
        out_x = enc_x.do_conv2d(weight, bias, windows_nb)

        return out_x

    # NOTE: This method requires private key access
    @staticmethod
    def backward(ctx: CKKSConvFunctionWrapper, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        # Get the saved tensors
        saved_tensors: Tuple[Tensor] = ctx.saved_tensors  # type: ignore
        x = ctx.enc_x.do_decryption()
        stride = ctx.stride

        # Unpack the saved tensors
        weight, = saved_tensors

        # Get the needs_input_grad
        result: Tuple[bool, bool, bool] = ctx.needs_input_grad  # type: ignore

        # Initialize the gradients
        grad_input = grad_weight = grad_bias = None

        if result[0]:
            # TODO: col2im the input
            # See: https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
            grad_input = conv2d_input(
                x.shape, weight, grad_output, stride=stride
            )
        if result[1]:
            # TODO: col2im the input
            # See: https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
            grad_weight = conv2d_weight(
                x, weight.shape, grad_output, stride=stride
            )
        if result[2]:
            grad_bias = grad_output.sum(0).reshape(1)

        return grad_input, grad_weight, grad_bias
