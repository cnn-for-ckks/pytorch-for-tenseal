from typing import Tuple, Optional
from torchseal.wrapper import CKKSWrapper
from torchseal.function import Conv2dFunction

import typing
import torch


class Conv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], input_size_with_channel: Tuple[int, int, int, int], stride: int = 1, padding: int = 0, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None) -> None:
        super(Conv2d, self).__init__()

        # Save the parameters
        self.input_size_with_channel = input_size_with_channel
        self.stride = stride
        self.padding = padding

        # Unpack the kernel size
        kernel_n_rows, kernel_n_cols = kernel_size

        # Create the weight and bias
        self.weight = torch.nn.Parameter(
            torch.rand(
                out_channels, in_channels, kernel_n_rows, kernel_n_cols
            ) if weight is None else weight
        )
        self.bias = torch.nn.Parameter(
            torch.rand(out_channels) if bias is None else bias
        )

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        out_x = typing.cast(
            CKKSWrapper,
            Conv2dFunction.apply(
                enc_x, self.weight, self.bias, self.input_size_with_channel, self.stride, self.padding
            )
        )

        return out_x
