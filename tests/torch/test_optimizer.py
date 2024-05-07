from typing import Iterator

import typing
import random
import numpy as np
import torch


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


def test_optimizer():
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

    # Declare the input dimensions
    batch_size = 3

    # Declare the model dimensions
    in_features = 3
    out_features = 2

    # Create the weight and bias tensors
    weight = torch.randn(out_features, in_features, requires_grad=True)
    bias = torch.randn(out_features, requires_grad=True)

    custom_weight = weight.clone().detach().requires_grad_(True)
    custom_bias = bias.clone().detach().requires_grad_(True)

    # Create the input tensor
    input = torch.randn(batch_size, in_features, requires_grad=True)
    custom_input = input.clone().detach().requires_grad_(True)

    # Create the target tensor
    target = torch.randn(batch_size, out_features)

    # Create the model
    model = torch.nn.Linear(in_features, out_features)
    model.weight = torch.nn.Parameter(weight)
    model.bias = torch.nn.Parameter(bias)

    custom_model = torch.nn.Linear(in_features, out_features)
    custom_model.weight = torch.nn.Parameter(custom_weight)
    custom_model.bias = torch.nn.Parameter(custom_bias)

    # Create the loss function
    criterion = torch.nn.MSELoss()
    custom_criterion = torch.nn.MSELoss()

    # Create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    custom_optimizer = SGD(custom_model.parameters(), lr=0.1)

    # Set the training mode
    model.train()
    custom_model.train()

    # Forward pass
    output = model.forward(input)
    custom_output = custom_model.forward(custom_input)

    # Calculate the loss
    loss = criterion.forward(output, target)
    custom_loss = custom_criterion.forward(custom_output, target)

    # Set the gradients to none
    optimizer.zero_grad()
    custom_optimizer.zero_grad()

    # Backward pass
    loss.backward()
    custom_loss.backward()

    # Update the weights
    optimizer.step()
    custom_optimizer.step()

    # Compare the weights
    assert torch.allclose(
        custom_model.weight,
        model.weight,
        atol=1e-3,
        rtol=0
    ), "Weights are not equal"
