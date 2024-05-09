import torch


def get_sparse_target(target: torch.Tensor, batch_size: int, num_classes: int) -> torch.Tensor:
    sparse_target = torch.zeros(
        batch_size, num_classes, dtype=torch.long
    ).scatter(1, target.unsqueeze(1), 1)

    return sparse_target
