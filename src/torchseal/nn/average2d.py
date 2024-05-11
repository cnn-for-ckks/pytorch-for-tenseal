from typing import Tuple
from torchseal.wrapper import CKKSWrapper
from torchseal.function import AvgPool2dFunction
from torchseal.utils import generate_near_zeros, approximate_toeplitz_multiple_channels, create_average_kernel, create_padding_transformation_matrix, create_inverse_padding_transformation_matrix

import typing
import torch


class AvgPool2d(torch.nn.Module):
    def __init__(self, n_channels: int, kernel_size: Tuple[int, int], input_size: Tuple[int, int], stride: int = 1, padding: int = 0) -> None:
        super(AvgPool2d, self).__init__()

        # Unpack the kernel size
        kernel_height, kernel_width = kernel_size

        # Unpack the input size
        input_height, input_width = input_size

        # Create the weight matrix for the average pooling operation
        self.weight = approximate_toeplitz_multiple_channels(
            create_average_kernel(
                n_channels, kernel_height, kernel_width
            ),
            (n_channels, input_height, input_width),
            stride=stride,
            padding=padding
        )

        # Create the binary masking for inference
        self.conv2d_padding_transformation = create_padding_transformation_matrix(
            n_channels, input_height, input_width, padding
        )

        self.conv2d_inverse_padding_transformation = create_inverse_padding_transformation_matrix(
            n_channels, input_height, input_width, padding
        )

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        enc_output = typing.cast(
            CKKSWrapper,
            AvgPool2dFunction.apply(
                enc_x, self.weight, self.conv2d_padding_transformation, self.conv2d_inverse_padding_transformation
            )
        )

        return enc_output
