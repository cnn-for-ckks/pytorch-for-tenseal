from typing import Tuple
from torch import Tensor
from torch.nn.functional import unfold
from tenseal import CKKSVector

import tenseal as ts


def im2col_encoding(context: ts.Context, image: Tensor, kernel_size: Tuple[int, int], stride: int, padding: int) -> Tuple[CKKSVector, int, int]:
    # Unfold the image
    unfolded_image = unfold(
        image,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # Encrypt the unfolded image
    enc_unfolded_image = ts.enc_matmul_encoding(
        context, unfolded_image.t()
    )

    # Get the number of rows and columns
    num_row = unfolded_image.shape[0]
    num_col = unfolded_image.shape[1]

    return enc_unfolded_image, num_row, num_col
