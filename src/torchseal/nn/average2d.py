from typing import Tuple
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.function import AvgPool2dFunction

import torch


# BUG: There is a bug (ValueError: encrypted1 and encrypted2 parameter mismatch)
class AvgPool2d(torch.nn.Module):
    def __init__(self, n_channel: int, kernel_size: Tuple[int, int], output_size: torch.Size, stride: int = 1, padding: int = 0,) -> None:
        super(AvgPool2d, self).__init__()

        # Save the parameters
        self.output_size = output_size
        self.stride = stride
        self.padding = padding

        # Unpack the kernel size
        kernel_n_rows, kernel_n_cols = kernel_size

        # Create the average kernel
        self.avg_kernel = torch.ones(
            n_channel, n_channel, kernel_n_rows, kernel_n_cols
        ).div(
            kernel_n_cols * kernel_n_rows
        )

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        out_x: CKKSWrapper = AvgPool2dFunction.apply(
            enc_x, self.avg_kernel, self.output_size, self.stride, self.padding
        )  # type: ignore

        return out_x
