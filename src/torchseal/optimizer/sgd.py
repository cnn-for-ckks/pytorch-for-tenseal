from typing import Iterator

import typing
import torch


# TODO: Move the SGD optimizer to the torchseal package
class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterator[torch.nn.Parameter], lr=1e-3) -> None:
        super(SGD, self).__init__(params, defaults={"lr": lr})

    def step(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                # Cast the parameter to a tensor
                param_tensor = typing.cast(torch.Tensor, param)

                # If the parameter does not require gradients, skip it
                if param_tensor.grad is None:
                    continue

                # Get the parameter
                lr = typing.cast(float, group["lr"])

                # Update the parameter
                param_tensor.data = param_tensor.data.subtract(
                    param_tensor.grad.data.mul(lr)
                )
