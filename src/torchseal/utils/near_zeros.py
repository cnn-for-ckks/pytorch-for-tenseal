from typing import Sequence, Union

import torch


# This function generate a near-zero tensor with the given shape and scale
# Useful for initializing toeplitz matrices
# NOTE: This is done to solve the possibility of parameter mismatch
def generate_near_zeros(shape: Union[int, Sequence[int]], scale: float = 1e-3) -> torch.Tensor:
    # Randomized with range [0, scale)
    return torch.rand(shape) * scale
