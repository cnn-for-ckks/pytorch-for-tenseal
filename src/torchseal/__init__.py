from torchseal.wrapper import CKKSWrapper
from torchseal.state import CKKSState

import torch
import tenseal as ts


def set_context(context: ts.Context) -> None:
    # Get the state of the CKKS
    state = CKKSState()

    # Set the context
    state.context = context


def ckks_wrapper(data: torch.Tensor, do_encryption: bool = True) -> CKKSWrapper:
    # Get the state of the CKKS
    state = CKKSState()

    # Create the ckks data
    ckks_data = ts.ckks_tensor(state.context, data.tolist())

    # Create the ckks wrapper
    instance = CKKSWrapper(data)

    # Set the ckks data
    instance.ckks_data = ckks_data

    # If encryption is enabled, encrypt the data
    if do_encryption:
        return instance.encrypt()

    # Else, return the instance without encryption
    return instance
