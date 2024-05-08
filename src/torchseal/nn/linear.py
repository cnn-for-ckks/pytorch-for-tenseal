from typing import Optional
from torchseal.wrapper import CKKSWrapper
from torchseal.function import LinearFunction

import typing
import torch


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None):
        super(Linear, self).__init__()

        self.weight = torch.nn.Parameter(
            torch.rand(out_features, in_features) if weight is None else weight
        )
        self.bias = torch.nn.Parameter(
            torch.rand(out_features) if bias is None else bias
        )

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        enc_output = typing.cast(
            CKKSWrapper,
            LinearFunction.apply(
                enc_x, self.weight, self.bias
            )
        )

        return enc_output
