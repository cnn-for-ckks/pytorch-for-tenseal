from typing import List

import torch


# NOTE: This is done to handle the case when the batch size is 1 (TenSEAL errors out in this case)
def create_empty_tensors(size: List[int]) -> torch.Tensor:
    if len(size) == 0:
        return torch.zeros(1)

    return torch.zeros(size)
