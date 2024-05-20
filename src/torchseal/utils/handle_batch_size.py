from typing import List

import torch


def create_empty_tensors(size: List[int]) -> torch.Tensor:
    if len(size) == 0:
        return torch.zeros(1)

    return torch.zeros(size)
