import tenseal as ts
import torch


# Function to compute the multiplicative inverse of an encrypted value using TenSEAL
def compute_multiplicative_inverse(context: ts.Context, encrypted_value: ts.CKKSTensor, iterations=5):
    # Start with an initial guess (encoded as a scalar)
    # For multiplicative inverse, good initial guess can be crucial - let's assume approx. inverse is known
    # This should be based on some estimation method
    inverse = ts.ckks_tensor(
        context, torch.ones(
            encrypted_value.shape
        ).tolist()
    )

    # Newton-Raphson iteration to refine the inverse
    for _ in range(iterations):
        prod = encrypted_value.mul(inverse)  # d * x_n

        correction = ts.ckks_tensor(
            context, torch.ones(
                encrypted_value.shape
            ).mul(2).tolist()
        ).add(
            prod.neg()
        )  # 2 - d * x_n

        inverse = inverse.mul(correction)  # x_n * (2 - d * x_n)

    return inverse


if __name__ == "__main__":
    # Controls precision of the fractional part
    bits_scale = 26

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[
            31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31
        ]
    )

    # Set the scale
    context.global_scale = pow(2, bits_scale)

    # Galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    # Example: Encrypted value of 0.5
    encrypted_value = ts.ckks_tensor(
        context, [1.0, 0.5, 0.25]
    )
    inverse_encrypted = compute_multiplicative_inverse(
        context, encrypted_value
    )

    # Decrypt to verify
    result = inverse_encrypted.decrypt().tolist()
    print("Decoded result:", result)
