from typing import Tuple, Optional
from torch import Tensor
from torch.nn import Module, Parameter
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.function.conv2d import Conv2dFunction

import torch


class Conv2d(Module):  # TODO: Add support for in_channels (this enables the use of multiple convolutions in a row)
    def __init__(self, kernel_size: Tuple[int, int], weight: Optional[Tensor] = None, bias: Optional[Tensor] = None) -> None:
        super(Conv2d, self).__init__()

        # Unpack the kernel size
        kernel_n_rows, kernel_n_cols = kernel_size

        # Create the weight and bias
        self.weight = Parameter(
            torch.rand(
                kernel_n_rows, kernel_n_cols
            )
        ) if weight is None else weight
        self.bias = Parameter(
            torch.rand(1)
        ) if bias is None else bias

    def forward(self, enc_x: CKKSWrapper, windows_nb: int) -> CKKSWrapper:
        out_x: CKKSWrapper = Conv2dFunction.apply(
            enc_x, self.weight, self.bias, windows_nb
        )  # type: ignore

        return out_x
