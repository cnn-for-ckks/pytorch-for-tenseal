from typing import Tuple
from torch import Tensor
from torch.nn.functional import fold, unfold
from tenseal import CKKSVector

import torch


def im2col_decoding(enc_unfolded_image: CKKSVector, num_row: int, num_col: int, output_size: Tuple[int, int], kernel_size: Tuple[int, int], stride: int, padding: int) -> Tensor:
    # Decrypt the result
    raw_dec_unfolded_image = enc_unfolded_image.decrypt()
    dec_unfolded_image = torch.tensor(raw_dec_unfolded_image).reshape(
        len(raw_dec_unfolded_image) // num_col, num_col
    )

    # Throw away the extra rows
    dec_unfolded_image_clipped = dec_unfolded_image[:num_row, :]

    # Fold
    dec_image = fold(
        dec_unfolded_image_clipped,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # Adjustment tensor
    adjustment_tensor = fold(
        unfold(
            torch.ones_like(dec_image),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ),
        output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding
    )

    # Adjust the result
    adjusted_dec_image = dec_image.div(adjustment_tensor)

    return adjusted_dec_image
