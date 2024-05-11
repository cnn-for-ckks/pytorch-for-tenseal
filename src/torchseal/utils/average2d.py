from torchseal.utils import generate_near_zeros

import torch


def create_average_kernel(n_channels: int, kernel_height: int, kernel_width: int) -> torch.Tensor:
    # Create the initial weight
    initial_weight = generate_near_zeros(
        (n_channels, n_channels, kernel_height, kernel_width)
    )

    # Fill the initial weight with the average pooling kernel
    for i in range(n_channels):
        initial_weight[i, i] = torch.ones(kernel_height, kernel_width).div(
            kernel_height * kernel_width
        )

    return initial_weight
