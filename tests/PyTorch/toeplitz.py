import torch


# Source: https://stackoverflow.com/questions/68896578/pytorchs-torch-as-strided-with-negative-strides-for-making-a-toeplitz-matrix/68899386#68899386
def toeplitz(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T

    return vals[j - i].reshape(*shape)


def toeplitz_one_channel(kernel: torch.Tensor, input_size: torch.Size) -> torch.Tensor:
    # Get the shapes
    kernel_height, kernel_width = kernel.shape
    input_height, input_width = input_size
    output_height = input_height - kernel_height + 1

    # Construct 1D convolution toeplitz matrices for each row of the kernel
    results = torch.stack([
        toeplitz(
            c=torch.stack([
                kernel[r, 0], *torch.zeros(input_width - kernel_width)
            ]), r=torch.stack([
                *kernel[r], *torch.zeros(input_width-kernel_width)
            ])
        ) for r in range(kernel_height)
    ])

    # Construct toeplitz matrix of toeplitz matrices (just for padding=0)
    number_of_blocks_height, number_of_blocks_width = output_height, input_height
    block_height, block_width = results[0].shape

    # Initialize the output tensor
    weight_convolutions = torch.zeros(
        (
            number_of_blocks_height,
            block_height,
            number_of_blocks_width,
            block_width
        )
    )

    # Fill the output tensor
    for i, block in enumerate(results):
        for j in range(output_height):
            weight_convolutions[j, :, i + j, :] = block

    # Reshape the output tensor
    weight_convolutions = weight_convolutions.view(
        number_of_blocks_height * block_height,
        number_of_blocks_width * block_width
    )

    return weight_convolutions


def toeplitz_multiple_channels(kernel: torch.Tensor, input_size: torch.Size) -> torch.Tensor:
    # Get the shapes
    kernel_height, kernel_width, kernel_channel, _ = kernel.shape
    input_height, input_width, input_channel = input_size
    output_size = torch.Size([
        kernel_height,
        input_width - kernel_width + 1,
        input_channel - kernel_channel + 1
    ])

    # Initialize the output tensor
    weight_convolutions = torch.zeros(
        (
            output_size[0],
            int(torch.prod(torch.tensor(output_size[1:])).item()),
            input_height,
            int(torch.prod(torch.tensor(input_size[1:])).item())
        )
    )

    # Fill the output tensor
    for i, kernel_output in enumerate(kernel):
        for j, kernel_input in enumerate(kernel_output):
            weight_convolutions[i, :, j, :] = toeplitz_one_channel(
                kernel_input, input_size[1:]
            )

    # Reshape the output tensor
    weight_convolutions = weight_convolutions.view(
        int(torch.prod(torch.tensor(output_size)).item()),
        int(torch.prod(torch.tensor(input_size)).item())
    )

    return weight_convolutions


if __name__ == "__main__":
    kernel = torch.randn(4, 3, 3, 3)
    input_tensor = torch.randn(3, 7, 9)

    toeplitz_matrix = toeplitz_multiple_channels(kernel, input_tensor.shape)
    output = toeplitz_matrix.matmul(input_tensor.view(-1)).view(1, 4, 5, 7)

    # Check the correctness of the convolution via the toeplitz matrix
    print(
        torch.sum(
            (output - torch.nn.functional.conv2d(
                input_tensor.view(1, 3, 7, 9), kernel
            ))**2
        ).item()
    )
