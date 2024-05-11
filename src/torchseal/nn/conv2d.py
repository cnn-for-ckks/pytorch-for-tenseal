from typing import Tuple, Optional
from torchseal.wrapper import CKKSWrapper
from torchseal.function.eval import Conv2dFunction
from torchseal.utils import approximate_toeplitz_multiple_channels, create_conv2d_input_mask, create_conv2d_weight_mask, create_conv2d_bias_transformation

import typing
import torch


class Conv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], input_size: Tuple[int, int], batch_size: int = 1, stride: int = 1, padding: int = 0, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None) -> None:
        super(Conv2d, self).__init__()

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

        # Create the weight and bias
        self.weight = torch.nn.Parameter(
            approximate_toeplitz_multiple_channels(
                torch.randn(
                    out_channels, in_channels, kernel_height, kernel_width
                ),
                (in_channels, input_height, input_width),
                stride=stride,
                padding=padding
            ) if weight is None else weight
        )

        self.bias = torch.nn.Parameter(
            torch.repeat_interleave(
                torch.randn(out_channels),
                output_height * output_width
            ) if bias is None else bias
        )

        # Create the binary masking for training
        self.conv2d_input_mask = create_conv2d_input_mask(
            (in_channels, input_height, input_width),
            batch_size=batch_size,
            padding=padding
        )

        self.conv2d_weight_mask = create_conv2d_weight_mask(
            (in_channels, input_height, input_width),
            (out_channels, in_channels, kernel_height, kernel_width),
            stride=stride,
            padding=padding
        )

        self.conv2d_bias_transformation = create_conv2d_bias_transformation(
            repeat=output_height * output_width,
            length=out_channels * output_height * output_width
        )

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        # TODO: Implement the forward pass based on self.training flag

        enc_output = typing.cast(
            CKKSWrapper,
            Conv2dFunction.apply(
                enc_x, self.weight, self.bias, self.conv2d_input_mask, self.conv2d_weight_mask, self.conv2d_bias_transformation
            )
        )

        return enc_output

    def train(self, mode=True) -> "Conv2d":
        # TODO: Change the plaintext parameters to encrypted parameters if mode is True
        # TODO: Else, change the encrypted parameters to plaintext parameters

        return super(Conv2d, self).train(mode)

    def eval(self) -> "Conv2d":
        # TODO: Change the encrypted parameters to plaintext parameters

        return super(Conv2d, self).eval()
