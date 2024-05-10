from torchseal.wrapper import CKKSWrapper

import torch
import tenseal as ts


def ckks_wrapper(context: ts.Context, data: torch.Tensor) -> CKKSWrapper:
    ckks_data = ts.ckks_tensor(context, data.tolist())
    ckks_data_wrapper = CKKSWrapper(ckks_data)

    return ckks_data_wrapper
