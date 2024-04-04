from typing import Tuple, Optional
from torch import Tensor
from torch.nn import Module, Parameter
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.function.conv2d import Conv2dFunction

import torch


class Conv2d(Module):  # TODO: Add support for in_channels and out_channels (this enables the use of multiple convolutions in a row)
    def __init__(self, in_channel: int, out_channel: int, kernel_size: Tuple[int, int], output_size: torch.Size, stride: int = 1, padding: int = 1, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None) -> None:
        super(Conv2d, self).__init__()

        # Save the parameters
        self.output_size = output_size
        self.stride = stride
        self.padding = padding

        # Unpack the kernel size
        kernel_n_rows, kernel_n_cols = kernel_size

        # Create the weight and bias
        self.weight = Parameter(
            torch.rand(
                out_channel, in_channel, kernel_n_rows, kernel_n_cols
            ) if weight is None else weight
        )
        self.bias = Parameter(
            torch.rand(out_channel) if bias is None else bias
        )

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        out_x: CKKSWrapper = Conv2dFunction.apply(
            enc_x, self.weight, self.bias, self.output_size, self.stride, self.padding
        )  # type: ignore

        return out_x
