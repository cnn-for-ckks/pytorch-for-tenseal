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
        context, [1, 0.8, 0.5]
    )
    inverse_encrypted = compute_multiplicative_inverse(
        encrypted_value, scale=1
    )

    # Decrypt to verify
    result = torch.tensor(inverse_encrypted.decrypt().tolist())
    target = torch.tensor([1, 1 / 0.8, 1 / 0.5])

    print(result)

    assert torch.allclose(result, target, atol=1e-1, rtol=0)
