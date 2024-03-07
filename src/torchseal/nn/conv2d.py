from typing import Tuple, Optional
from torch import Tensor
from torch.nn import Module, Parameter
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.function.conv2d import Conv2dFunction

import torch


class Conv2d(Module):  # TODO: Add support for in_channels and out_channels (this enables the use of multiple convolutions in a row)
    def __init__(self, output_size: Tuple[int, int], kernel_size: Tuple[int, int], stride: int, padding: int, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None) -> None:
        super(Conv2d, self).__init__()

        # Save the parameters
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Unpack the kernel size
        kernel_n_rows, kernel_n_cols = self.kernel_size

        # Create the weight and bias
        self.weight = Parameter(
            torch.rand(
                1, 1, kernel_n_rows, kernel_n_cols
            ) if weight is None else weight
        )
        self.bias = Parameter(
            torch.rand(1) if bias is None else bias
        )

    def forward(self, enc_x: CKKSWrapper, num_row: int, num_col: int) -> CKKSWrapper:
        out_x: CKKSWrapper = Conv2dFunction.apply(
            enc_x, self.weight, self.bias, num_row, num_col, self.output_size, self.kernel_size, self.stride, self.padding
        )  # type: ignore

        return out_x
