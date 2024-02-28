from torchseal import Linear

import torch
import tenseal as ts


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
