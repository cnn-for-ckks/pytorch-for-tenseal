from typing import Tuple, Optional
from torch import Tensor
from torch.nn import Module, Parameter
from tenseal import CKKSVector
from torchseal.function.conv2d import Conv2dFunction

import torch


class Conv2d(Module):  # TODO: Add support for in_channels (this enables the use of multiple convolutions in a row)
    def __init__(self, out_channels: int, kernel_size: Tuple[int, int], weight: Optional[Tensor] = None, bias: Optional[Tensor] = None) -> None:
        super(Conv2d, self).__init__()

        # Unpack the kernel size
        kernel_n_rows, kernel_n_cols = kernel_size

        # Create the weight and bias
        self.weight = Parameter(
            torch.rand(
                out_channels, kernel_n_rows, kernel_n_cols
            )
        ) if weight is None else weight
        self.bias = Parameter(
            torch.rand(out_channels)
        ) if bias is None else bias

    def forward(self, enc_x: CKKSVector, windows_nb: int) -> CKKSVector:
        result: CKKSVector = Conv2dFunction.apply(
            enc_x, self.weight, self.bias, windows_nb
        )  # type: ignore

        return result
