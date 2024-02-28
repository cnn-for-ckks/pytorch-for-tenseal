from typing import List
from torch.nn import Module
from tenseal import CKKSVector
import numpy as np
import tenseal as ts


class Linear(Module):
    weight: List[List[float]]
    bias: List[float]

    def __init__(self, num_input: int, num_output: int):
        super(Linear, self).__init__()

        self.weight = [
            [
                np.random.random_sample() for _ in range(num_output)
            ]
            for _ in range(num_input)
        ]
        self.bias = [np.random.random_sample() for _ in range(num_output)]

    def forward(self, enc_x: CKKSVector) -> CKKSVector:
        return enc_x.matmul(self.weight).add(self.bias)


if __name__ == "__main__":
    # Seed random number generator
    np.random.seed(0)

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
    enc_vec = ts.ckks_vector(context, [1, 2, 3])

    # Get the output
    output = model.forward(enc_vec)

    # Decrypt the output
    print(output.decrypt())
