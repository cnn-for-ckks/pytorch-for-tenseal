from typing import Tuple
from torch import Tensor
from torch.autograd import Function
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.wrapper.function import CKKSFunctionWrapper


class SquareFunction(Function):
    @staticmethod
    def forward(ctx: CKKSFunctionWrapper, enc_x: CKKSWrapper) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.enc_x = enc_x

        # Apply square function to the encrypted input
        out_x = enc_x.do_square()

        return out_x

    @staticmethod
    def backward(ctx: CKKSFunctionWrapper, grad_output: Tensor) -> Tuple[Tensor]:
        # Get the saved tensors
        enc_x = ctx.enc_x

        # Do the backward operation
        out_x = enc_x.do_square_backward()

        # Compute the gradients
        grad_input = grad_output.mm(out_x)

        return grad_input,
