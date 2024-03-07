from torch.nn.functional import unfold, fold

import torch
import numpy as np
import random

if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

    N, C, H, W = 1, 1, 6, 6  # Batch size, Channels, Height, Width
    x = torch.arange(N * C * H * W).view(N, C, H, W).float()

    # Parameters for unfold
    # BUG: Must not overlap
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)  # Adding padding to see its effect

    # Unfold
    unfolded = unfold(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # Fold (Inverse operation)
    folded = fold(
        unfolded,
        output_size=(H, W),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # Check if the folded tensor is equal to the original tensor
    print("Is x equal to folded?", torch.allclose(x, folded))
