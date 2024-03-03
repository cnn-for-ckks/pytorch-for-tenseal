from typing import Optional
from torch import Tensor
from torch.nn import Module, Parameter
from tenseal import CKKSVector
from torchseal.function.linear import LinearFunction

import torch


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None):
        super(Linear, self).__init__()

        self.weight = Parameter(
            torch.rand(in_features, out_features)
        ) if weight is None else weight
        self.bias = Parameter(
            torch.rand(out_features)
        ) if bias is None else bias

    def forward(self, enc_x: CKKSVector) -> CKKSVector:
        result: CKKSVector = LinearFunction.apply(
            enc_x, self.weight, self.bias
        )  # type: ignore

        return result
