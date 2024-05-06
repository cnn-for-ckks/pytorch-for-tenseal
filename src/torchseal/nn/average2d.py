from typing import Tuple
from torchseal.wrapper import CKKSWrapper
from torchseal.function import AvgPool2dFunction
from torchseal.utils import approximate_toeplitz_multiple_channels

import typing
import torch


class AvgPool2d(torch.nn.Module):
    def __init__(self, n_channel: int, kernel_size: Tuple[int, int], input_size_with_channel: Tuple[int, int, int, int], stride: int = 1, padding: int = 0,) -> None:
        super(AvgPool2d, self).__init__()

        # Save the parameters
        self.input_size_with_channel = input_size_with_channel
        self.stride = stride
        self.padding = padding

        # Unpack the kernel size
        kernel_n_rows, kernel_n_cols = kernel_size

        # Create the average kernel
        self.avg_kernel = torch.ones(
            n_channel, n_channel, kernel_n_rows, kernel_n_cols
        ).div(
            kernel_n_cols * kernel_n_rows
        )
        self.toeplitz_avg_kernel = approximate_toeplitz_multiple_channels(
            self.avg_kernel, input_size_with_channel[1:], stride=stride, padding=padding
        )

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        out_x = typing.cast(
            CKKSWrapper,
            AvgPool2dFunction.apply(
                enc_x, self.avg_kernel, self.toeplitz_avg_kernel, self.input_size_with_channel, self.stride, self.padding
            )
        )

        return out_x
