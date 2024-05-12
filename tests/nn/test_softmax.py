from torchseal.nn import Softmax as EncryptedSoftmax
from torch.nn import Softmax as PlainSoftmax

import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_softmax():
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

    # Controls precision of the fractional part
    bits_scale = 26

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[
            31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31
        ]
    )

    # Set the scale
    context.global_scale = pow(2, bits_scale)

    # Galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    # Declare parameters
    exp_start = -3
    exp_stop = 3
    exp_num_of_sample = 100
    exp_degree = 4
    exp_approximation_type = "minimax"

    inverse_start = 10
    inverse_stop = 30
    inverse_num_of_sample = 100
    inverse_degree = 2
    inverse_approximation_type = "minimax"
    inverse_iterations = 3

    # Declare input dimensions
    batch_size = 1
    input_length = 10

    # Create the input tensor
    input_tensor = torch.randn(batch_size, input_length)

    # Encrypt the value
    enc_input_tensor = torchseal.ckks_wrapper(
        context, input_tensor
    )

    # Create the plaintext softmax layer
    softmax = PlainSoftmax(dim=1)

    # Create the encrypted softmax layer
    encrypted_softmax = EncryptedSoftmax(
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
        inverse_iterations
    )

    # Calculate the output
    output = softmax.forward(input_tensor)
    enc_output = encrypted_softmax.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.do_decryption()

    # Check the correctness of the convolution (with a tolerance of 5e-2)
    assert torch.allclose(
        dec_output, output, atol=5e-2, rtol=0
    ), "Softmax layer failed!"

    # TODO: Do backward pass and check the correctness of the input gradients
