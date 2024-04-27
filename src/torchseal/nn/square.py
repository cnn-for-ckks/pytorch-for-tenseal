from torchseal.function import SquareFunction
from torchseal.wrapper import CKKSWrapper

import typing
import torch


class Square(torch.nn.Module):
    def __init__(self) -> None:
        super(Square, self).__init__()

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        out_x = typing.cast(
            CKKSWrapper,
            SquareFunction.apply(
                enc_x
            )
        )

        return out_x
