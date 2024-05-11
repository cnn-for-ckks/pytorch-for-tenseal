from torchseal.wrapper import CKKSWrapper
from torchseal.function.eval import SquareFunction

import typing
import torch


class Square(torch.nn.Module):
    def __init__(self) -> None:
        super(Square, self).__init__()

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        # TODO: Implement the forward pass based on self.training flag

        enc_output = typing.cast(
            CKKSWrapper,
            SquareFunction.apply(
                enc_x
            )
        )

        return enc_output

    def train(self, mode=True) -> "Square":
        # TODO: Change the plaintext parameters to encrypted parameters if mode is True
        # TODO: Else, change the encrypted parameters to plaintext parameters

        return super(Square, self).train(mode)

    def eval(self) -> "Square":
        # TODO: Change the encrypted parameters to plaintext parameters

        return super(Square, self).eval()
