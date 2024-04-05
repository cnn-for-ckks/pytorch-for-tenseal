from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.utils import near_zeros

import torch
import tenseal as ts


def ckks_wrapper(context: ts.Context, data: torch.Tensor) -> CKKSWrapper:
    enc_data = ts.ckks_tensor(context, data.tolist())
    enc_data_wrapper = CKKSWrapper(
        near_zeros(enc_data.shape), enc_data
    )
    return enc_data_wrapper
