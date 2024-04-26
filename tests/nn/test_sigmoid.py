from torchseal.nn import Sigmoid as EncryptedSigmoid
from torch.nn import Sigmoid as PlainSigmoid

import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_sigmoid():
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

    # Declare parameters
    start = -3
    stop = 3
    num_of_sample = 100
    degree = 3
    approximation_type = "minimax"

    # Declare input dimensions
    input_length = 10
    batch_size = 1

    # Create the input tensor
    input_tensor = torch.randn(batch_size, input_length)

    # Encrypt the input tensor
    enc_input_tensor = torchseal.ckks_wrapper(
        context, input_tensor.view(batch_size, -1)
    )

    # Create the plaintext sigmoid layer
    sigmoid = PlainSigmoid()

    # Create the encrypted sigmoid layer
    enc_sigmoid = EncryptedSigmoid(
        start=start,
        stop=stop,
        num_of_sample=num_of_sample,
        degree=degree,
        approximation_type=approximation_type,
    )

    # Calculate the output
    output = sigmoid.forward(input_tensor)
    enc_output = enc_sigmoid.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.do_decryption()

    # Check the correctness of the convolution (with a tolerance of 5e-2)
    assert torch.allclose(
        output, dec_output, atol=5e-2, rtol=0
    ), "Sigmoid layer failed!"


# TODO: Add gradient test
