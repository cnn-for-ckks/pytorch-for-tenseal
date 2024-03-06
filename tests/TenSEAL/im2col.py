from typing import Tuple
from tenseal import CKKSVector

import tenseal as ts
import torch

if __name__ == "__main__":
    # Controls precision of the fractional part
    bits_scale = 26

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[
            31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31
        ]
    )

    # Set the scale
    context.global_scale = pow(2, bits_scale)

    # Galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    # Load the data
    raw_data = torch.rand(28 * 28)

    # Create the im2col encoding
    # NOTE: Somehow this function pads the kernel to the nearest power of 2
    result: Tuple[CKKSVector, int] = ts.im2col_encoding(
        context,
        raw_data.view(28, 28).tolist(),
        5,
        5,
        1
    )  # type: ignore
    enc_x, windows_nb = result

    # Print the length decrypted result
    print(
        "Output shape: [{}, {}]".format(
            len(enc_x.decrypt()) // windows_nb, windows_nb
        )
    )
