from torch.nn import Module
from torchseal.function import SquareFunction
from torchseal.wrapper.ckks import CKKSWrapper


class Square(Module):
    def __init__(self) -> None:
        super(Square, self).__init__()

    def forward(self, enc_x: CKKSWrapper) -> CKKSWrapper:
        out_x: CKKSWrapper = SquareFunction.apply(
            enc_x
        )  # type: ignore

        return out_x
