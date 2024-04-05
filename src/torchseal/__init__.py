from torchseal.wrapper.ckks import CKKSWrapper

import torch
import tenseal as ts


def ckks_wrapper(context: ts.Context, data: torch.Tensor) -> CKKSWrapper:
    enc_data = ts.ckks_vector(context, data.tolist())
    enc_data_wrapper = CKKSWrapper(
        torch.zeros(enc_data.size()), enc_data
    )
    return enc_data_wrapper
