from typing import Optional
from torchseal.wrapper import CKKSWrapper
from torchseal.function import LinearFunction

import typing
import torch
import torchseal


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, weight: Optional[CKKSWrapper] = None, bias: Optional[CKKSWrapper] = None):
        super(Linear, self).__init__()

        self.weight = typing.cast(
            CKKSWrapper,
            torch.nn.Parameter(
                torchseal.ckks_wrapper(
                    torch.rand(out_features, in_features),
                    do_encryption=True
                ) if weight is None else weight
            )
        )

        self.bias = typing.cast(
            CKKSWrapper,
            torch.nn.Parameter(
                torchseal.ckks_wrapper(
                    torch.rand(out_features),
                    do_encryption=True
                ) if bias is None else bias
            )
        )

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        enc_output = typing.cast(
            CKKSWrapper,
            LinearFunction.apply(
                enc_x, self.weight, self.bias, self.training
            )
        )

        return enc_output

    def train(self, mode=True) -> "Linear":
        if mode:
            # Inplace encrypt the parameters
            self.weight.inplace_encrypt()
            self.bias.inplace_encrypt()
        else:
            # Inplace decrypt the parameters
            self.weight.inplace_decrypt()
            self.bias.inplace_decrypt()

        return super(Linear, self).train(mode)

    def eval(self) -> "Linear":
        return self.train(False)
