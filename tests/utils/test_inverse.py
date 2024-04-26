from numpy.polynomial import Polynomial

import numpy as np
import torch
import tenseal as ts
import torchseal


def test_compute_multiplicative_inverse():
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

    # Create the polynomial
    start = 0.5
    stop = 1.0
    num_of_sample = 5

    x = np.linspace(start, stop, num_of_sample)
    y = (lambda x: 1 / x)(x)
    degree = 1

    # Compute the polynomial approximation
    polyval_coeffs: np.ndarray = Polynomial.fit(
        x, y, degree
    ).convert(kind=Polynomial).coef

    # Encrypt the value
    encrypted_value = torchseal.ckks_wrapper(
        context, torch.tensor([1, 0.8, 0.5])
    )

    # Compute the multiplicative inverse
    iterations = 4
    encrypted_value.do_multiplicative_inverse(polyval_coeffs, iterations)

    # Decrypt to verify
    result = encrypted_value.do_decryption()
    target = torch.tensor([1, 1 / 0.8, 1 / 0.5])

    print(result)

    assert torch.allclose(
        result, target, atol=1e-1, rtol=0
    ), "Inverse computation failed!"
