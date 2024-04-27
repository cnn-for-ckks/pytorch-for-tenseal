from numpy.polynomial import Polynomial

import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_compute_multiplicative_inverse():
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
    start = 0.5
    stop = 1.0
    num_of_sample = 5
    degree = 3
    iterations = 4

    # Declare input dimensions
    input_length = 10
    batch_size = 1

    # Calculate the sample points
    x = np.linspace(start, stop, num_of_sample)
    y = (lambda x: 1 / x)(x)

    # Fit the polynomial
    polyval_coeffs: np.ndarray = Polynomial.fit(
        x, y, degree
    ).convert(kind=Polynomial).coef

    # Create the input tensor
    input_tensor = start + (stop - start) * torch.rand(
        batch_size, input_length
    )

    # Encrypt the value
    enc_input_tensor = torchseal.ckks_wrapper(
        context, input_tensor
    )

    # Compute the multiplicative inverse
    enc_input_tensor = enc_input_tensor.do_multiplicative_inverse(
        polyval_coeffs, iterations
    )

    # Decrypt to verify
    result = enc_input_tensor.do_decryption()
    target = input_tensor.pow(-1)

    # Check the correctness of the convolution (with a tolerance of 5e-2)
    assert torch.allclose(
        result, target, atol=5e-2, rtol=0
    ), "Inverse operation failed!"
