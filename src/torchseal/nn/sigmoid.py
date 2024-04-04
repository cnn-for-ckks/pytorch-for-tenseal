from torchseal.function import SigmoidFunction
from torchseal.wrapper.ckks import CKKSWrapper

import torch


class Sigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super(Sigmoid, self).__init__()

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        out_x: CKKSWrapper = SigmoidFunction.apply(
            enc_x
        )  # type: ignore

        return out_x
