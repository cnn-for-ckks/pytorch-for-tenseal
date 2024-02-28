from typing import Tuple
from torch.nn import Module
from tenseal import CKKSVector, PlainTensor
import torch
import tenseal as ts


class Conv2d(Module):  # TODO: Add support for channels and bias
    kernel: PlainTensor

    def __init__(self, kernel_n_rows: int, kernel_n_cols: int) -> None:
        super(Conv2d, self).__init__()

        self.kernel = ts.plain_tensor(
            torch.rand(kernel_n_rows, kernel_n_cols).tolist(),
            [kernel_n_rows, kernel_n_cols]
        )

    def forward(self, enc_x: CKKSVector, windows_nb: int) -> CKKSVector:
        return enc_x.conv2d_im2col(self.kernel, windows_nb)


if __name__ == "__main__":
    # Seed random number generator
    torch.manual_seed(0)

    # Define the parameter
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
    model = Conv2d(kernel_n_rows, kernel_n_cols)

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
