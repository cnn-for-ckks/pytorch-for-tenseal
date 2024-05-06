from typing import Optional, Tuple
from torch.autograd.function import NestedIOFunction

import typing
import random
import numpy as np
import torch


class CrossEntropyLossFunctionWrapper(NestedIOFunction):
    pass


class CrossEntropyLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CrossEntropyLossFunctionWrapper, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply softmax to the input to get probabilities (will be approximated for encrypted data)
        output = torch.nn.functional.softmax(input, dim=1)

        # Apply log softmax to the output (will be approximated for encrypted data)
        log_output = torch.log(output)

        # Calculate the loss (negative log likelihood)
        # TODO: Model this so that it can be used on encrypted data
        loss = torch.nn.functional.nll_loss(log_output, target)

        # Save input and target for backward pass
        ctx.save_for_backward(output, target)

        return loss

    @staticmethod
    def backward(ctx: CrossEntropyLossFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        # Retrieve saved tensors
        output, target = typing.cast(
            Tuple[torch.Tensor, torch.Tensor],
            ctx.saved_tensors
        )

        # Unpack the target shape
        batch_size, = target.shape

        # Create the subtraction mask
        # TODO: Model this so that it can be used on encrypted data
        subtraction_mask = torch.zeros_like(
            output
        ).scatter(1, target.unsqueeze(1), -1)

        # Add the subtraction mask to the output
        output_subtracted = output.add(subtraction_mask)

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[
                bool, bool
            ],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        grad_input = None

        if result[0]:
            # Multiply by the grad_output
            grad_input = grad_output.mul(output_subtracted).div(batch_size)

        return grad_input, None


class CrossEntropyLoss(torch.nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return typing.cast(
            torch.Tensor,
            CrossEntropyLossFunction.apply(input, target)
        )


def test_cross_entropy():
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

    # Declare input dimensions
    batch_size = 3
    num_classes = 5

    # Create the input tensors
    input = torch.randn(batch_size, num_classes, requires_grad=True)
    custom_input = input.clone().detach().requires_grad_(True)

    # Create the target tensor
    target = torch.randint(
        high=num_classes,
        size=(batch_size, ),
    )

    # Instantiate the custom function
    loss_layer = torch.nn.CrossEntropyLoss()
    custom_loss_layer = CrossEntropyLoss()

    # Compute the loss and gradients
    loss = loss_layer.forward(input, target)
    custom_loss = custom_loss_layer.forward(custom_input, target)

    # Check the correctness of the results (with a tolerance of 1e-3)
    assert torch.allclose(
        custom_loss,
        loss,
        atol=1e-3,
        rtol=0
    ), "Losses are not equal"

    # Compute the gradients
    loss.backward()
    custom_loss.backward()

    # Check the correctness of the gradients (with a tolerance of 1e-3)
    assert input.grad is not None and custom_input.grad is not None, "Gradients are None"

    assert torch.allclose(
        custom_input.grad,
        input.grad,
        atol=1e-3,
        rtol=0
    ), "Gradients are not equal"
