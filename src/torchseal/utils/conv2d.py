from typing import Tuple
from torchseal.utils import approximate_toeplitz_multiple_channels, generate_near_zeros

import numpy as np
import torch


def create_conv2d_weight_mask(input_size_with_channel: Tuple[int, int, int], kernel_size_with_channel: Tuple[int, int, int, int], stride: int = 1, padding: int = 0) -> torch.Tensor:
    # Get the shapes
    out_channels, _, kernel_height, kernel_width = kernel_size_with_channel
    in_channels, input_height, input_width = input_size_with_channel

    # Calculate the length of the index tensor
    index_length = out_channels * in_channels * kernel_height * kernel_width

    # Create the index tensor
    index_tensor = approximate_toeplitz_multiple_channels(
        torch.arange(1, index_length + 1).view(
            out_channels, in_channels, kernel_height, kernel_width
        ),
        (in_channels, input_height, input_width),
        stride=stride,
        padding=padding
    )

    # Create the binary tensor
    binary_tensor = torch.stack([
        torch.tensor(
            np.vectorize(
                lambda x: 1.000 if np.isclose(x, i, atol=1e-3, rtol=0) else generate_near_zeros(
                    (1, 1)
                ).item()
            )(index_tensor)
        )
        for i in range(1, index_length + 1)
    ])

    return binary_tensor


def create_conv2d_bias_transformation(repeat: int, length: int) -> torch.Tensor:
    # Calculate the number of groups
    num_groups = length // repeat

    # Create the transformation matrix
    transformation_matrix = generate_near_zeros((length, length))

    # Fill the transformation matrix
    for i in range(num_groups):
        for j in range(repeat):
            row = i * repeat + j
            group_start = i * repeat
            transformation_matrix[row, group_start:group_start + repeat] = 1

    return transformation_matrix
