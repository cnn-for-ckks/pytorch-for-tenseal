from torch.nn import Module
from tenseal import CKKSVector, PlainTensor
import torch
import tenseal as ts


class Linear(Module):
    weight: PlainTensor
    bias: PlainTensor

    def __init__(self, in_features: int, out_features: int):
        super(Linear, self).__init__()

        self.weight = ts.plain_tensor(
            torch.rand(in_features, out_features).tolist(),
            [in_features, out_features]
        )
        self.bias = ts.plain_tensor(
            torch.rand(out_features).tolist(),
            [out_features]
        )

    def forward(self, enc_x: CKKSVector) -> CKKSVector:
        return enc_x.matmul(self.weight).add(self.bias)


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
