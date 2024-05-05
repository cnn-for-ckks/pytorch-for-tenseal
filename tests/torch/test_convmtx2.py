from typing import Tuple, Optional
from torch.nn import Conv2d
from torch.autograd.function import NestedIOFunction

import typing
import torch
import numpy as np
import random


# Source: https://stackoverflow.com/questions/68896578/pytorchs-torch-as-strided-with-negative-strides-for-making-a-toeplitz-matrix/68899386#68899386
def toeplitz(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T

    return vals[j - i].view(*shape)


# Source: https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n
def toeplitz_one_channel(kernel: torch.Tensor, input_size: Tuple[int, int], stride: int = 1, padding: int = 0) -> torch.Tensor:
    kernel_height, kernel_width = kernel.shape
    input_height, input_width = input_size

    # Adjust the input dimensions based on padding
    padded_input_height = input_height + 2 * padding
    padded_input_width = input_width + 2 * padding

    # Calculate the output dimensions considering stride and padding
    output_height = ((padded_input_height - kernel_height) // stride) + 1

    # Stack and reshape the matrices to form the final weight matrix
    toeplitz_matrices = torch.stack([
        # Construct 1D convolution toeplitz matrices for each row of the kernel considering stride
        toeplitz(
            torch.cat(
                [
                    kernel[r, 0:1],
                    torch.zeros(
                        padded_input_height - kernel_height
                    )
                ]
            ),
            torch.cat(
                [kernel[r, :], torch.zeros(padded_input_width - kernel_width)]
            )
        )[::stride, :][:output_height]
        for r in range(kernel_height)
    ])

    # Calculate the number of blocks and their sizes for constructing the final matrix
    num_blocks_height = output_height
    block_height, block_width = toeplitz_matrices[0].shape

    # Initialize the final weight matrix with zeros
    weight_matrix = torch.zeros(
        (num_blocks_height * block_height, padded_input_width * padded_input_height)
    )

    # Fill in the blocks for the final weight matrix
    for i in range(kernel_height):
        for j in range(output_height):
            start_row = j * block_height
            end_row = start_row + block_height
            start_col = (i + j * stride) * padded_input_width
            end_col = start_col + block_width
            weight_matrix[
                start_row:end_row, start_col:end_col
            ] = toeplitz_matrices[i]

    return weight_matrix


# Source: https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n
def toeplitz_multiple_channels(kernel: torch.Tensor, input_size_with_channel: Tuple[int, int, int], stride: int = 1, padding: int = 0) -> torch.Tensor:
    # Get the shapes
    kernel_out_channel, _, kernel_height, kernel_width = kernel.shape
    input_in_channel, input_height, input_width = input_size_with_channel

    # Adjust for padding
    padded_input_height = input_height + 2 * padding
    padded_input_width = input_width + 2 * padding

    # Calculate the output size (with padding and stride)
    output_height = (padded_input_height - kernel_height) // stride + 1
    output_width = (padded_input_width - kernel_width) // stride + 1

    # Initialize the output tensor
    weight_convolutions = torch.zeros(
        (
            kernel_out_channel,
            output_height * output_width,
            input_in_channel,
            padded_input_height * padded_input_width
        )
    )

    # Fill the output tensor
    for i, kernel_output in enumerate(kernel):
        for j, kernel_single_channel in enumerate(kernel_output):
            weight_convolutions[i, :, j, :] = toeplitz_one_channel(
                kernel_single_channel,
                (input_height, input_width),
                padding=padding,
                stride=stride
            )

    # Reshape the output tensor
    weight_convolutions = weight_convolutions.view(
        kernel_out_channel * output_height * output_width,
        input_in_channel * padded_input_height * padded_input_width
    )

    return weight_convolutions


def create_transformation_matrix(repeat: int, length: int):
    # Calculate the number of groups
    num_groups = length // repeat

    # Create the transformation matrix
    transformation_matrix = torch.zeros((length, length))

    for i in range(num_groups):
        for j in range(repeat):
            row = i * repeat + j
            group_start = i * repeat
            transformation_matrix[row, group_start:group_start + repeat] = 1

    return transformation_matrix


class ToeplitzConv2dFunctionWrapper(NestedIOFunction):
    stride: int
    padding: int
    input_size_with_channel: Tuple[int, int, int]
    kernel_size_with_channel: Tuple[int, int, int, int]


class ToeplitzConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: ToeplitzConv2dFunctionWrapper, padded_x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, input_size_with_channel: Tuple[int, int, int], kernel_size_with_channel: Tuple[int, int, int, int]) -> torch.Tensor:
        # Save the context for the backward method
        ctx.save_for_backward(padded_x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.input_size_with_channel = input_size_with_channel
        ctx.kernel_size_with_channel = kernel_size_with_channel

        # Apply the linear transformation to the input
        out_x = padded_x.mm(weight.t()).add(bias)

        return out_x

    @staticmethod
    def backward(ctx: ToeplitzConv2dFunctionWrapper, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Get the saved tensors
        padded_x, weight = typing.cast(
            Tuple[torch.Tensor, torch.Tensor],
            ctx.saved_tensors
        )

        # Get the saved context
        stride = ctx.stride
        padding = ctx.padding
        input_size_with_channel = ctx.input_size_with_channel
        kernel_size_with_channel = ctx.kernel_size_with_channel

        # Unpack the padded input size
        batch_size, _ = padded_x.shape

        # Get the shapes
        out_channels, _, kernel_height, kernel_width = kernel_size_with_channel
        in_channels, input_height, input_width = input_size_with_channel

        # Add padding to the input size
        padded_input_height = input_height + 2 * padding
        padded_input_width = input_width + 2 * padding

        # Count the output dimensions
        output_height = (padded_input_height - kernel_height) // stride + 1
        output_width = (padded_input_width - kernel_width) // stride + 1

        # Get the needs_input_grad
        result = typing.cast(
            Tuple[
                bool, bool, bool, bool, bool, bool, bool
            ],
            ctx.needs_input_grad
        )

        # Initialize the gradients
        grad_input = grad_weight = grad_bias = None

        # Compute the gradients
        if result[0]:
            # Create binary mask to remove padding
            binary_mask = torch.nn.functional.pad(
                torch.ones(
                    in_channels, input_height, input_width
                ),
                (padding, padding, padding, padding),
            ).view(-1).unsqueeze(0).repeat(batch_size, 1)

            # Calculate the gradients for the input tensor (this will be encrypted)
            grad_input = grad_output.mm(weight).mul(binary_mask)

        if result[1]:
            # Calculate the length of the index tensor
            index_length = out_channels * in_channels * kernel_height * kernel_width

            # Create the index tensor
            index_tensor = toeplitz_multiple_channels(
                torch.arange(1, index_length + 1).view(
                    out_channels, in_channels, kernel_height, kernel_width
                ),
                (in_channels, input_height, input_width),
                stride=stride,
                padding=padding
            )

            # Create the binary tensor
            binary_tensor = torch.stack([
                torch.tensor(
                    np.vectorize(
                        lambda x: x == i
                    )(index_tensor)
                )
                for i in range(1, index_length + 1)
            ])

            # Create the fully connected gradient weight tensor (this will be encrypted)
            unfiltered_grad_weight = grad_output.t().mm(padded_x)

            # Initialize the gradient weight tensor (this will be encrypted, probably going to need context with public keys)
            grad_weight = torch.zeros_like(weight)

            # Apply the binary tensor to the gradient weight (this will be encrypted)
            for binary_mask in binary_tensor:
                # Apply the binary mask to the gradient weight (this will be encrypted)
                filtered_grad_weight = unfiltered_grad_weight.mul(binary_mask)

                # Calculate the sum of all elements (this will be encrypted)
                sum_all_element = filtered_grad_weight.sum(1).sum(0)

                # Create the new current gradient weight (this will be encrypted)
                current_gradient_weight = binary_mask.mul(sum_all_element)

                # Add the current gradient weight to the final gradient weight (this will be encrypted)
                grad_weight += current_gradient_weight

        if result[2]:
            binary_transformation = create_transformation_matrix(
                repeat=output_height * output_width,
                length=out_channels * output_height * output_width
            )

            grad_bias = binary_transformation.matmul(grad_output.sum(0))

        return grad_input, grad_weight, grad_bias, None, None, None, None


class ToeplitzConv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], input_size: Tuple[int, int], stride: int = 1, padding: int = 0) -> None:
        super().__init__()

        # Unpack the kernel size
        kernel_height, kernel_width = kernel_size

        # Unpack the input size
        input_height, input_width = input_size

        # Adjust for padding
        padded_input_height = input_height + 2 * padding
        padded_input_width = input_width + 2 * padding

        # Count the output dimensions
        output_height = (padded_input_height - kernel_height) // stride + 1
        output_width = (padded_input_width - kernel_width) // stride + 1

        # Save the parameters
        self.stride = stride
        self.padding = padding
        self.input_size_with_channel = (in_channels, input_height, input_width)
        self.kernel_size_with_channel = (
            out_channels, in_channels, kernel_height, kernel_width
        )

        # Create the weight and bias
        self.weight = torch.nn.Parameter(
            toeplitz_multiple_channels(
                torch.randn(
                    out_channels, in_channels, kernel_height, kernel_width
                ),
                self.input_size_with_channel,
                stride=stride,
                padding=padding
            )
        )

        self.bias = torch.nn.Parameter(
            torch.repeat_interleave(
                torch.randn(out_channels),
                output_height * output_width
            )
        )

    def forward(self, padded_x: torch.Tensor) -> torch.Tensor:
        out_x = typing.cast(
            torch.Tensor,
            ToeplitzConv2dFunction.apply(
                padded_x, self.weight, self.bias, self.stride, self.padding, self.input_size_with_channel, self.kernel_size_with_channel
            )
        )

        return out_x


def test_convmtx2():
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

    # Declare parameters
    out_channels = 3
    in_channels = 2
    kernel_height = 3
    kernel_width = 3
    stride = 1
    padding = 1

    # Declare input dimensions
    batch_size = 1
    input_height = 5
    input_width = 5

    # Adjust for padding
    padded_input_height = input_height + 2 * padding
    padded_input_width = input_width + 2 * padding

    # Count the output dimensions
    output_height = (padded_input_height - kernel_height) // stride + 1
    output_width = (padded_input_width - kernel_width) // stride + 1

    # Create weight and bias
    kernel = torch.randn(
        out_channels,
        in_channels,
        kernel_height,
        kernel_width
    )
    bias = torch.randn(out_channels)

    # Create the sparse kernel
    sparse_kernel = toeplitz_multiple_channels(
        kernel,
        (in_channels, input_height, input_width),
        stride=stride,
        padding=padding
    )
    sparse_bias = bias.repeat_interleave(output_height * output_width)

    # Create the input tensor
    input_tensor = torch.randn(
        batch_size, in_channels, input_height, input_width, requires_grad=True
    )

    # Create the sparse input tensor (with padding)
    sparse_input_tensor = torch.nn.functional.pad(
        input_tensor, (padding, padding, padding, padding)
    ).view(
        batch_size, -1
    ).clone().detach().requires_grad_(True)

    # Create the convolution layer
    conv2d = Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_height, kernel_width),
        stride=stride,
        padding=padding,
        bias=True,
    )
    conv2d.weight = torch.nn.Parameter(kernel)
    conv2d.bias = torch.nn.Parameter(bias)

    # Create the sparse convolution layer
    sparse_conv2d = ToeplitzConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_height, kernel_width),
        input_size=(input_height, input_width),
        stride=stride,
        padding=padding,
    )
    sparse_conv2d.weight = torch.nn.Parameter(sparse_kernel)
    sparse_conv2d.bias = torch.nn.Parameter(sparse_bias)

    # Perform the forward pass
    output_conv2d = conv2d.forward(input_tensor)
    output_sparse_conv2d = sparse_conv2d.forward(sparse_input_tensor)

    # Check the correctness of the convolution (with a tolerance of 1e-3)
    output_conv2d_expanded = output_conv2d.view(batch_size, -1)

    assert torch.allclose(
        output_sparse_conv2d,
        output_conv2d_expanded,
        atol=1e-3,
        rtol=0
    ), "Convmtx2 forward pass is incorrect!"

    # Define the target
    target = torch.randn(batch_size, out_channels, output_height, output_width)

    # Define the criterion
    criterion = torch.nn.L1Loss()

    # Calculate the loss
    loss_conv2d = criterion.forward(output_conv2d, target)
    loss_sparse_conv2d = criterion.forward(
        output_sparse_conv2d, target.view(batch_size, -1)
    )

    # Perform the backward pass
    loss_conv2d.backward()
    loss_sparse_conv2d.backward()

    # Check the correctness of the input gradients (with a tolerance of 1e-3)
    assert input_tensor.grad is not None and sparse_input_tensor.grad is not None, "Input gradients are None!"

    input_tensor_grad_expanded = torch.nn.functional.pad(
        input_tensor.grad, (padding, padding, padding, padding)
    ).view(batch_size, -1)

    assert torch.allclose(
        sparse_input_tensor.grad,
        input_tensor_grad_expanded,
        atol=1e-3,
        rtol=0
    ), "Input gradients are incorrect!"

    # Check the correctness of the weight gradients (with a tolerance of 1e-3)
    assert conv2d.weight.grad is not None and sparse_conv2d.weight.grad is not None, "Weight gradients are None!"

    conv2d_weight_grad_expanded = toeplitz_multiple_channels(
        conv2d.weight.grad,
        (in_channels, input_height, input_width),
        stride=stride,
        padding=padding
    )

    assert torch.allclose(
        sparse_conv2d.weight.grad,
        conv2d_weight_grad_expanded,
        atol=1e-3,
        rtol=0
    ), "Weight gradients are incorrect!"

    # Check the correctness of the bias gradients (with a tolerance of 1e-3)
    assert conv2d.bias.grad is not None and sparse_conv2d.bias.grad is not None, "Bias gradients are None!"

    conv2d_bias_grad_expanded = conv2d.bias.grad.repeat_interleave(
        output_height * output_width
    )

    assert torch.allclose(
        sparse_conv2d.bias.grad,
        conv2d_bias_grad_expanded,
        atol=1e-3,
        rtol=0
    ), "Bias gradients are incorrect!"
