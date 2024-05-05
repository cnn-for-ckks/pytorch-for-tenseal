import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_sum():
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
            31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31
        ]
    )

    # Set the scale
    context.global_scale = pow(2, bits_scale)

    # Galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    # Declare parameters
    axis = 1

    # Declare input dimensions
    input_length = 10
    batch_size = 1

    # Create the input tensor
    input_tensor = torch.randn(batch_size, input_length)

    # Encrypt the value
    enc_input_tensor = torchseal.ckks_wrapper(
        context, input_tensor
    )

    # Do the sum
    enc_input_tensor = enc_input_tensor.do_sum(axis=axis)

    # Decrypt to verify
    result = enc_input_tensor.do_decryption()
    target = input_tensor.sum(axis)

    # Check the correctness of the convolution (with a tolerance of 5e-2)
    assert torch.allclose(
        result, target, atol=5e-2, rtol=0
    ), "Sum operation failed!"
