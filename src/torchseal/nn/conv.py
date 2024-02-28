from typing import List, Tuple
from torch.nn import Module
from tenseal import CKKSVector, PlainTensor
import torch
import tenseal as ts


class Conv2d(Module):  # TODO: Add support for in_channels
    weight: PlainTensor
    bias: PlainTensor

    def __init__(self, out_channels: int, kernel_size: Tuple[int, int]) -> None:
        super(Conv2d, self).__init__()

        # Unpack the kernel size
        kernel_n_rows, kernel_n_cols = kernel_size

        # Create the weight and bias
        self.weight = ts.plain_tensor(
            torch.rand(
                out_channels,
                kernel_n_rows,
                kernel_n_cols
            ).tolist(),
            [out_channels, kernel_n_rows, kernel_n_cols]
        )
        self.bias = ts.plain_tensor(
            torch.rand(out_channels).tolist(),
            [out_channels]
        )

    def forward(self, enc_x: CKKSVector, windows_nb: int):
        list_of_weight: List[List[List[float]]] = self.weight.tolist()
        list_of_bias: List[float] = self.bias.tolist()

        return CKKSVector.pack_vectors([
            enc_x.conv2d_im2col(kernel, windows_nb).add(bias) for kernel, bias in zip(list_of_weight, list_of_bias)
        ])


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
