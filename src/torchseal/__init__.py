from torchseal.wrapper import CKKSWrapper
from torchseal.state import CKKSState

import torch
import tenseal as ts


def set_context(context: ts.Context) -> None:
    # Get the state of the CKKS
    state = CKKSState()

    # Set the context
    state.context = context


def ckks_wrapper(data: torch.Tensor, do_encryption: bool = False) -> CKKSWrapper:
    # Create the ckks wrapper
    instance = CKKSWrapper(data)

    # If encryption is enabled, encrypt the data
    if do_encryption:
        return instance.encrypt()

    # Else, return the instance without encryption
    return instance
