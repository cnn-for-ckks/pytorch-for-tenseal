from typing import Tuple
from torchseal.wrapper import CKKSWrapper
from torchseal.function import AvgPool2dFunction
from torchseal.utils import approximate_toeplitz_multiple_channels, create_conv2d_input_mask

import typing
import torch


class AvgPool2d(torch.nn.Module):
    def __init__(self, n_channels: int, kernel_size: Tuple[int, int], input_size: Tuple[int, int], batch_size: int = 1, stride: int = 1, padding: int = 0,) -> None:
        super(AvgPool2d, self).__init__()

        # Unpack the kernel size
        kernel_height, kernel_width = kernel_size

        # Unpack the input size
        input_height, input_width = input_size

        # Create the weight
        self.weight = approximate_toeplitz_multiple_channels(
            torch.ones(
                n_channels, n_channels, kernel_height, kernel_width
            ).div(
                kernel_height * kernel_width
            ),
            (n_channels, input_height, input_width),
            stride=stride,
            padding=padding
        )

        # Create the binary masking for training
        self.conv2d_input_mask = create_conv2d_input_mask(
            (n_channels, input_height, input_width),
            batch_size=batch_size,
            padding=padding
        )

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        enc_output = typing.cast(
            CKKSWrapper,
            AvgPool2dFunction.apply(
                enc_x, self.weight, self.conv2d_input_mask
            )
        )

        return enc_output
