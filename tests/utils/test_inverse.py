from torchseal.utils import compute_multiplicative_inverse

import tenseal as ts
import torch


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

    # Example: Encrypted value
    encrypted_value = ts.ckks_tensor(
        context, [100, 80, 50]
    )
    inverse_encrypted = compute_multiplicative_inverse(
        encrypted_value, scale=100
    )

    # Decrypt to verify
    result = torch.tensor(inverse_encrypted.decrypt().tolist())
    target = torch.tensor([1/100, 1/80, 1/50])

    assert torch.allclose(result, target, atol=1e-2, rtol=0)
