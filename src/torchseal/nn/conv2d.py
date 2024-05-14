from typing import Tuple, Optional

from torchseal.wrapper import CKKSWrapper
from torchseal.function import Conv2dFunction
from torchseal.utils import approximate_toeplitz_multiple_channels, create_conv2d_weight_mask, create_conv2d_bias_transformation, create_padding_transformation_matrix, create_inverse_padding_transformation_matrix

import typing
import torch
import torchseal


class Conv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], input_size: Tuple[int, int], stride: int = 1, padding: int = 0, weight: Optional[CKKSWrapper] = None, bias: Optional[CKKSWrapper] = None) -> None:
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
        self.weight = typing.cast(
            CKKSWrapper,
            torch.nn.Parameter(
                torchseal.ckks_wrapper(
                    approximate_toeplitz_multiple_channels(
                        torch.randn(
                            out_channels, in_channels, kernel_height, kernel_width
                        ),
                        (in_channels, input_height, input_width),
                        stride=stride,
                        padding=padding
                    ),
                    do_encryption=True
                ) if weight is None else weight
            )
        )

        self.bias = typing.cast(
            CKKSWrapper,
            torch.nn.Parameter(
                torchseal.ckks_wrapper(
                    torch.repeat_interleave(
                        torch.randn(out_channels),
                        output_height * output_width
                    ),
                    do_encryption=True
                ) if bias is None else bias
            )
        )

        # Create the binary masking for inference
        self.conv2d_padding_transformation = create_padding_transformation_matrix(
            in_channels, input_height, input_width, padding
        )

        self.conv2d_inverse_padding_transformation = create_inverse_padding_transformation_matrix(
            out_channels, input_height, input_width, padding
        )

        # Create the binary masking for training
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
        enc_output = typing.cast(
            CKKSWrapper,
            Conv2dFunction.apply(
                enc_x, self.weight, self.bias, self.conv2d_padding_transformation, self.conv2d_inverse_padding_transformation, self.conv2d_weight_mask, self.conv2d_bias_transformation, self.training
            )
        )

        return enc_output

    def train(self, mode=True) -> "Conv2d":
        if mode:
            # Inplace encrypt the parameters
            self.weight.inplace_encrypt()
            self.bias.inplace_encrypt()
        else:
            # Inplace decrypt the parameters
            self.weight.inplace_decrypt()
            self.bias.inplace_decrypt()

        return super(Conv2d, self).train(mode)

    def eval(self) -> "Conv2d":
        return self.train(False)
