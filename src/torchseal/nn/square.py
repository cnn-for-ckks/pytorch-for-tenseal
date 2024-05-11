from torchseal.wrapper import CKKSWrapper
from torchseal.function import SquareFunction

import typing
import torch


class Square(torch.nn.Module):
    def __init__(self) -> None:
        super(Square, self).__init__()

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        enc_output = typing.cast(
            CKKSWrapper,
            SquareFunction.apply(
                enc_x
            )
        )

        return enc_output
