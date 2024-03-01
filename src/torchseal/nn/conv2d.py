from typing import Tuple, Optional
from torch import Tensor
from torch.nn import Module, Parameter
from torch.autograd import Function
from torch.autograd.function import NestedIOFunction
from tenseal import CKKSVector

import torch


# NOTE: Unused at the moment
class Conv2dFunctionCtx(NestedIOFunction):
    enc_x: CKKSVector


# NOTE: Unused at the moment
class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx: Conv2dFunctionCtx, enc_x: CKKSVector, weight: Tensor, bias: Tensor, windows_nb: int) -> CKKSVector:
        # Save the ctx for the backward method
        ctx.save_for_backward(weight, bias)
        ctx.enc_x = enc_x

        # TODO: Move pack_vectors to the "Flatten" layer
        return CKKSVector.pack_vectors([
            enc_x.conv2d_im2col(kernel, windows_nb).add(bias) for kernel, bias in zip(weight.tolist(), bias.tolist())
        ])

    # TODO: Test the backward method
    # NOTE: This method requires private key access
    @staticmethod
    def backward(ctx: Conv2dFunctionCtx, enc_grad_output: CKKSVector) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        # Get the saved tensors
        weight, bias = ctx.saved_tensors
        enc_x = ctx.enc_x

        # Check if the weight and bias have the right type
        if type(weight) != torch.Tensor or type(bias) != torch.Tensor:
            raise TypeError("The weight and bias should be tensors")

        # Get the input gradient
        result: Tuple[bool, bool, bool] = ctx.needs_input_grad  # type: ignore

        # Decrypt the encrypted input and gradient
        x = torch.tensor(enc_x.decrypt(), requires_grad=True)
        grad_output = torch.tensor(
            enc_grad_output.decrypt(),
            requires_grad=True
        )

        # Initialize the gradients
        grad_input = grad_weight = grad_bias = None

        # TODO: Compute the gradients

        return grad_input, grad_weight, grad_bias

    @staticmethod
    def apply(enc_x: CKKSVector, weight: Tensor, bias: Tensor, windows_nb: int) -> CKKSVector:
        result = super(Conv2dFunction, Conv2dFunction).apply(
            enc_x, weight, bias, windows_nb
        )

        if type(result) != CKKSVector:
            raise TypeError("The result should be a CKKSVector")

        return result


class Conv2d(Module):  # TODO: Add support for in_channels (this enables the use of multiple convolutions in a row)
    weight: Tensor
    bias: Tensor

    def __init__(self, out_channels: int, kernel_size: Tuple[int, int], weight: Optional[Tensor] = None, bias: Optional[Tensor] = None) -> None:
        super(Conv2d, self).__init__()

        # Unpack the kernel size
        kernel_n_rows, kernel_n_cols = kernel_size

        # Create the weight and bias
        self.weight = Parameter(
            torch.rand(
                out_channels, kernel_n_rows, kernel_n_cols
            ),
            requires_grad=True
        ) if weight is None else weight
        self.bias = Parameter(
            torch.rand(out_channels),
            requires_grad=True
        ) if bias is None else bias

    def forward(self, enc_x: CKKSVector, windows_nb: int) -> CKKSVector:
        return Conv2dFunction.apply(enc_x, self.weight, self.bias, windows_nb)
