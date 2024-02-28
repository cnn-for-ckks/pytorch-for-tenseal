from typing import List, Tuple
from torch import Tensor
from torch.nn import Module, Parameter
from tenseal import CKKSVector

import torch
import tenseal as ts


class Conv2d(Module):  # TODO: Add support for in_channels (this enables the use of multiple convolutions in a row)
    weight: Tensor
    bias: Tensor

    def __init__(self, out_channels: int, kernel_size: Tuple[int, int]) -> None:
        super(Conv2d, self).__init__()

        # Unpack the kernel size
        kernel_n_rows, kernel_n_cols = kernel_size

        # Create the weight and bias
        self.weight = Parameter(
            torch.empty(
                out_channels, kernel_n_rows, kernel_n_cols
            )
        )
        self.bias = Parameter(torch.empty(out_channels))

    def forward(self, enc_x: CKKSVector, windows_nb: int):
        list_of_weight: List[List[List[float]]] = self.weight.tolist()
        list_of_bias: List[float] = self.bias.tolist()

        # TODO: Move pack_vectors to the "Flatten" layer
        return CKKSVector.pack_vectors([
            enc_x.conv2d_im2col(kernel, windows_nb).add(bias) for kernel, bias in zip(list_of_weight, list_of_bias)
        ])

    # TODO: Add the backward method to enable training


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
