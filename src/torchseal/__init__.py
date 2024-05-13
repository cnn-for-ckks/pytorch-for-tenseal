from torchseal.wrapper import CKKSWrapper
from torchseal.state import CKKSState

import torch
import tenseal as ts


def set_context(context: ts.Context) -> None:
    # Get the state of the CKKS
    state = CKKSState()

    # Set the context
    state.context = context


def ckks_wrapper(data: torch.Tensor) -> CKKSWrapper:
    return CKKSWrapper(data)
