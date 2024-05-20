from torchseal.nn import CrossEntropyLoss as EncryptedCrossEntropyLoss
from torchseal.utils import get_sparse_target
from torchseal.wrapper import CKKSWrapper
from torch.nn import CrossEntropyLoss as PlainCrossEntropyLoss

import typing
import math
import random
import numpy as np
import torch
import tenseal as ts
import torchseal


def test_cross_entropy():
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

    # Controls precision of the fractional part
    bits_scale = 26

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=32768,
        coeff_mod_bit_sizes=[
            31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31
        ]
    )

    # Set the scale
    context.global_scale = pow(2, bits_scale)

    # Galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    # Set the context
    torchseal.set_context(context)

    # Declare parameters
    exp_start = -3
    exp_stop = 3
    exp_num_of_sample = 100
    exp_degree = 4
    exp_approximation_type = "minimax"

    inverse_start = 2.5
    inverse_stop = 7.5
    inverse_num_of_sample = 100
    inverse_degree = 2
    inverse_approximation_type = "minimax"
    inverse_iterations = 3

    log_start = math.exp(-4)
    log_stop = 1
    log_num_of_sample = 100
    log_degree = 4
    log_approximation_type = "minimax"

    # Declare input dimensions
    batch_size = 2
    num_classes = 5

    # Create the input tensors
    input_tensor = torch.randn(batch_size, num_classes, requires_grad=True)

    # Encrypt the value
    enc_input_tensor = torchseal.ckks_wrapper(input_tensor, do_encryption=True)
    enc_input_tensor.requires_grad_(True)

    # Create the target tensor
    target = torch.randint(
        high=num_classes,
        size=(batch_size, ),
    )

    # Sparse and encrypt the target tensor
    enc_target = torchseal.ckks_wrapper(
        get_sparse_target(target, batch_size, num_classes), do_encryption=True
    )

    # Create the plaintext loss layer
    criterion = PlainCrossEntropyLoss()

    # Create the encrypted loss layer
    enc_criterion = EncryptedCrossEntropyLoss(
        exp_start,
        exp_stop,
        exp_num_of_sample,
        exp_degree,
        exp_approximation_type,
        inverse_start,
        inverse_stop,
        inverse_num_of_sample,
        inverse_degree,
        inverse_approximation_type,
        inverse_iterations,
        log_start,
        log_stop,
        log_num_of_sample,
        log_degree,
        log_approximation_type
    )

    # Calculate the output
    loss = criterion.forward(input_tensor, target)
    enc_loss = enc_criterion.forward(enc_input_tensor, enc_target)

    # Decrypt the output
    dec_loss = enc_loss.decrypt().plaintext_data.clone()

    # Check the correctness of the results (with a tolerance of 0.5, because the log function will expand the error)
    assert torch.allclose(
        dec_loss, loss, atol=0.5, rtol=0
    ), "Cross entropy loss layer failed!"

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
