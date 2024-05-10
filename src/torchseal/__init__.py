from torchseal.wrapper import CKKSWrapper

import torch
import tenseal as ts


def ckks_wrapper(context: ts.Context, data: torch.Tensor) -> CKKSWrapper:
    return CKKSWrapper(context, data)
