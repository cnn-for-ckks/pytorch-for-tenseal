from torchseal.utils import toeplitz_multiple_channels

import torch

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
