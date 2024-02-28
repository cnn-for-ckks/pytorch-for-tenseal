from torch import Tensor
from torch.nn import Module, Parameter
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
from tenseal import CKKSVector

import torch
import tenseal as ts


class LinearFunction(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, enc_x: CKKSVector, weight: Tensor, bias: Tensor) -> CKKSVector:
        # TODO: Save the ctx for the backward method

        # Unpack the weight and bias
        plain_weight = ts.plain_tensor(weight.tolist(), weight.shape)
        plain_bias = ts.plain_tensor(bias.tolist(), bias.shape)

        return enc_x.matmul(plain_weight).add(plain_bias)

    @staticmethod
    def apply(enc_x: CKKSVector, weight: Tensor, bias: Tensor) -> CKKSVector:
        result = super(LinearFunction, LinearFunction).apply(
            enc_x, weight, bias
        )

        if type(result) != CKKSVector:
            raise TypeError("The result should be a CKKSVector")

        return result

    # TODO: Define the backward method to enable training


class Linear(Module):
    weight: Tensor
    bias: Tensor

    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()

        self.weight = Parameter(torch.empty(in_features, out_features))
        self.bias = Parameter(torch.empty(out_features))

    def forward(self, enc_x: CKKSVector) -> CKKSVector:
        return LinearFunction.apply(enc_x, self.weight, self.bias)


if __name__ == "__main__":
    # Seed random number generator
    torch.manual_seed(0)

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[31, 26, 26, 26, 26, 26, 26, 31]
    )

    # Set the scale
    context.global_scale = pow(2, 26)

    # Galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    # Create a model
    model = Linear(3, 2)

    # Create a CKKSVector
    enc_vec = ts.ckks_vector(context, torch.rand(3).tolist())

    # Get the output
    output = model.forward(enc_vec)

    # Decrypt the output
    print(output.decrypt())
