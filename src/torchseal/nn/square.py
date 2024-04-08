from torchseal.function import SquareFunction
from torchseal.wrapper import CKKSWrapper

import torch


class Square(torch.nn.Module):
    def __init__(self) -> None:
        super(Square, self).__init__()

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        out_x: CKKSWrapper = SquareFunction.apply(
            enc_x
        )  # type: ignore

        return out_x
