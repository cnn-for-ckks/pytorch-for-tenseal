from typing import Iterator
from torchseal.wrapper import CKKSWrapper

import typing
import torch


# TODO: Move the SGD optimizer to operate on encrypted data
class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterator[torch.nn.Parameter], lr=1e-3) -> None:
        super(SGD, self).__init__(params, defaults={"lr": lr})

    def step(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                # If the parameter is not a ckks wrapper or not encrypted, skip it
                if not isinstance(param, CKKSWrapper) or not param.is_encrypted():
                    continue

                # If the parameter does not require gradients, a ckks wrapper, or not encrypted, skip it
                if param.grad is None or not isinstance(param.grad, CKKSWrapper) or not param.grad.is_encrypted():
                    continue

                # Get the parameter
                lr = typing.cast(float, group["lr"])

                # Delta weight
                delta_weight = param.grad.ckks_apply_scalar(-lr)

                # Update the tensor without creating a new one
                param.ckks_data = param.ckks_data.add(
                    delta_weight.ckks_data
                )
