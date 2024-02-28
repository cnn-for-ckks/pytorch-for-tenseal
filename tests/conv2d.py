from typing import Tuple
from tenseal import CKKSVector
from torchseal import Conv2d

import torch
import tenseal as ts


if __name__ == "__main__":
    # Seed random number generator
    torch.manual_seed(0)

    # Define the parameter
    out_channels = 4
    kernel_n_rows = 2
    kernel_n_cols = 2
    stride = 2

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
    model = Conv2d(out_channels, (kernel_n_rows, kernel_n_cols))

    # Create a CKKSVector from an image
    result: Tuple[CKKSVector, int] = ts.im2col_encoding(
        context,
        torch.rand(4, 4).tolist(),
        kernel_n_rows,
        kernel_n_cols,
        stride
    )  # type: ignore[error from tenseal library]

    # Unpack the result
    enc_vec, windows_nb = result

    # Get the output
    output = model.forward(enc_vec, windows_nb)

    # Decrypt the output
    print(output.decrypt())
