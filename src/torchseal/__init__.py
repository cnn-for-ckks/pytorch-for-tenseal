from torchseal.wrapper import CKKSWrapper

import torch
import tenseal as ts


def ckks_wrapper(context: ts.Context, data: torch.Tensor, is_encrypted: bool = True) -> CKKSWrapper:
    return CKKSWrapper(context, data, is_encrypted)
