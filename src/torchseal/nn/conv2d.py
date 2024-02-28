from typing import Tuple, Union
from torch import Tensor
from torch.nn import Module, Parameter
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from tenseal import CKKSVector

import torch


class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, enc_x: CKKSVector, weight: Tensor, bias: Tensor, windows_nb: int) -> CKKSVector:
        # TODO: Save the ctx for the backward method

        # TODO: Move pack_vectors to the "Flatten" layer
        return CKKSVector.pack_vectors([
            enc_x.conv2d_im2col(kernel, windows_nb).add(bias) for kernel, bias in zip(weight.tolist(), bias.tolist())
        ])

    @staticmethod
    def apply(enc_x: CKKSVector, weight: Tensor, bias: Tensor, windows_nb: int) -> CKKSVector:
        result = super(Conv2dFunction, Conv2dFunction).apply(
            enc_x, weight, bias, windows_nb
        )

        if type(result) != CKKSVector:
            raise TypeError("The result should be a CKKSVector")

        return result

    # TODO: Define the backward method to enable training


class Conv2d(Module):  # TODO: Add support for in_channels (this enables the use of multiple convolutions in a row)
    weight: Tensor
    bias: Tensor

    def __init__(self, out_channels: int, kernel_size: Tuple[int, int], weight: Union[Tensor, None] = None, bias: Union[Tensor, None] = None) -> None:
        super(Conv2d, self).__init__()

        # Unpack the kernel size
        kernel_n_rows, kernel_n_cols = kernel_size

        # Create the weight and bias
        self.weight = Parameter(
            torch.empty(
                out_channels, kernel_n_rows, kernel_n_cols
            )
        ) if weight is None else weight
        self.bias = Parameter(
            torch.empty(out_channels)
        ) if bias is None else bias

    def forward(self, enc_x: CKKSVector, windows_nb: int):
        return Conv2dFunction.apply(enc_x, self.weight, self.bias, windows_nb)
