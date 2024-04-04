import torch


# Source: https://stackoverflow.com/questions/68896578/pytorchs-torch-as-strided-with-negative-strides-for-making-a-toeplitz-matrix/68899386#68899386
def toeplitz(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T

    return vals[j - i].view(*shape)


# Source: https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n
def toeplitz_one_channel(kernel: torch.Tensor, input_size: torch.Size, stride: int = 1, padding: int = 0) -> torch.Tensor:
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
def toeplitz_multiple_channels(kernel: torch.Tensor, input_size: torch.Size, stride: int = 1, padding: int = 0) -> torch.Tensor:
    # Get the shapes
    kernel_out_channel, _, kernel_height, kernel_width = kernel.shape
    input_in_channel, input_height, input_width = input_size

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
            input_height * input_width
        )
    )

    # Fill the output tensor
    for i, kernel_output in enumerate(kernel):
        for j, kernel_input in enumerate(kernel_output):
            weight_convolutions[i, :, j, :] = toeplitz_one_channel(
                kernel_input, input_size[1:], padding=padding, stride=stride
            )

    # Reshape the output tensor
    weight_convolutions = weight_convolutions.view(
        kernel_out_channel * output_height * output_width,
        input_in_channel * input_height * input_width
    )

    return weight_convolutions
