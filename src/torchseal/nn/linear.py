from torch import Tensor
from torch.nn import Module, Parameter
from tenseal import CKKSVector

import torch
import tenseal as ts


class Linear(Module):
    weight: Tensor
    bias: Tensor

    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()

        self.weight = Parameter(torch.empty(in_features, out_features))
        self.bias = Parameter(torch.empty(out_features))

    def forward(self, enc_x: CKKSVector) -> CKKSVector:
        weight = ts.plain_tensor(self.weight.tolist(), self.weight.shape)
        bias = ts.plain_tensor(self.bias.tolist(), self.bias.shape)

        return enc_x.matmul(weight).add(bias)

    # TODO: Add the backward method to enable training


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
