from typing import Tuple
from torch import Tensor
from torch.autograd import Function
from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.wrapper.function import CKKSFunctionWrapper


class SquareFunction(Function):
    @staticmethod
    def forward(ctx: CKKSFunctionWrapper, enc_x: CKKSWrapper) -> CKKSWrapper:
        # Save the ctx for the backward method
        ctx.enc_x = enc_x.clone()

        # Apply square function to the encrypted input
        out_x = enc_x.do_square()

        return out_x

    @staticmethod
    def backward(ctx: CKKSFunctionWrapper, grad_output: Tensor) -> Tuple[Tensor]:
        # Get the saved tensors
        x = ctx.enc_x.do_decryption()

        # Do the backward operation
        out = x.do_square_backward()

        # Compute the gradients
        grad_input = grad_output.matmul(out)

        return grad_input,
