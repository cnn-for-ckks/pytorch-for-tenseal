from torchseal.wrapper import CKKSWrapper
from torchseal.function import MSELossFunction

import typing
import torch


class MSELoss(torch.nn.Module):
    def __init__(self) -> None:
        super(MSELoss, self).__init__()

    def forward(self, input: CKKSWrapper, target: CKKSWrapper) -> CKKSWrapper:
        enc_loss = typing.cast(
            CKKSWrapper,
            MSELossFunction.apply(input, target),
        )

        return enc_loss
