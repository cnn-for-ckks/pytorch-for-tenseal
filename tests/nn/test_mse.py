from torchseal.nn import MSELoss as EncryptedMSELoss
from torchseal.wrapper import CKKSWrapper
from torch.nn import MSELoss as PlainMSELoss

import typing
import random
import numpy as np
import torch
import tenseal as ts
import torchseal


def test_mse():
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

    # Controls precision of the fractional part
    bits_scale = 26

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[
            31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31
        ]
    )

    # Set the scale
    context.global_scale = pow(2, bits_scale)

    # Galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    # Set the context
    torchseal.set_context(context)

    # Declare input dimensions
    batch_size = 2
    num_classes = 5

    # Create the input tensors
    input_tensor = torch.randn(batch_size, num_classes, requires_grad=True)

    # Encrypt the value
    enc_input_tensor = torchseal.ckks_wrapper(input_tensor, do_encryption=True)
    enc_input_tensor.requires_grad_(True)

    # Create the target tensor
    target = torch.randn(batch_size, num_classes)

    # Sparse and encrypt the target tensor
    enc_target = torchseal.ckks_wrapper(
        target.clone(), do_encryption=True
    )

    # Create the plaintext loss layer
    criterion = PlainMSELoss()

    # Create the encrypted loss layer
    enc_criterion = EncryptedMSELoss()

    # Calculate the output
    loss = criterion.forward(input_tensor, target)
    enc_loss = enc_criterion.forward(enc_input_tensor, enc_target)

    # Decrypt the output
    dec_loss = enc_loss.decrypt().plaintext_data.clone()

    # Check the correctness of the results (with a tolerance of 5e-2)
    assert torch.allclose(
        dec_loss, loss, atol=5e-2, rtol=0
    ), "Mean squared error loss layer failed!"

    # Do backward pass
    loss.backward()
    enc_loss.backward()

    # Check the correctness of input gradients (with a tolerance of 0.5)
    assert enc_input_tensor.grad is not None and input_tensor.grad is not None, "Input gradients are None!"

    # Decrypt the input gradients
    dec_input_grad = typing.cast(
        CKKSWrapper,
        enc_input_tensor.grad
    ).decrypt().plaintext_data.clone()

    assert torch.allclose(
        dec_input_grad,
        input_tensor.grad,
        atol=0.5,
        rtol=0
    ), "Input gradients are incorrect!"
