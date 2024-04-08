from typing import Tuple, Optional
from torchseal.wrapper import CKKSWrapper
from torchseal.function import Conv2dFunction

import torch


class Conv2d(torch.nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: Tuple[int, int], input_size: torch.Size, stride: int = 1, padding: int = 0, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None) -> None:
        super(Conv2d, self).__init__()

        # Save the parameters
        self.input_size = input_size
        self.stride = stride
        self.padding = padding

        # Unpack the kernel size
        kernel_n_rows, kernel_n_cols = kernel_size

        # Create the weight and bias
        self.weight = torch.nn.Parameter(
            torch.rand(
                out_channel, in_channel, kernel_n_rows, kernel_n_cols
            ) if weight is None else weight
        )
        self.bias = torch.nn.Parameter(
            torch.rand(out_channel) if bias is None else bias
        )

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        out_x: CKKSWrapper = Conv2dFunction.apply(
            enc_x, self.weight, self.bias, self.input_size, self.stride, self.padding
        )  # type: ignore

        return out_x
