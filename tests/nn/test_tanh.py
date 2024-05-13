from torchseal.nn import Tanh as EncryptedTanh
from torch.nn import Tanh as PlainTanh

import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_tanh():
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

    # Declare parameters
    start = -3
    stop = 3
    num_of_sample = 100
    degree = 5
    approximation_type = "minimax"

    # Declare input dimensions
    input_length = 10
    batch_size = 1

    # Create the input tensor
    input_tensor = torch.randn(batch_size, input_length)

    # Encrypt the input tensor
    enc_input_tensor = torchseal.ckks_wrapper(
        input_tensor.view(batch_size, -1)
    )

    # Create the plaintext tanh layer
    tanh = PlainTanh()

    # Create the encrypted tanh layer
    enc_tanh = EncryptedTanh(
        start=start,
        stop=stop,
        num_of_sample=num_of_sample,
        degree=degree,
        approximation_type=approximation_type,
    )

    # Calculate the output
    output = tanh.forward(input_tensor)
    enc_output = enc_tanh.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.decrypt()

    # Check the correctness of the convolution (with a tolerance of 5e-2)
    assert torch.allclose(
        output, dec_output, atol=5e-2, rtol=0
    ), "Tanh layer failed!"

    # TODO: Do backward pass and check the correctness of the input gradients
