from torch import Tensor
from tenseal import CKKSVector


def im2col_conv2d(enc_unfolded_image: CKKSVector, conv_weight: Tensor, num_col: int) -> CKKSVector:
    # Do Encrypted convolution
    enc_result = enc_unfolded_image.enc_matmul_plain(
        conv_weight.tolist(), num_col
    )

    return enc_result
