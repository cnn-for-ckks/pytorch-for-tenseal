import tenseal as ts
import torch
import math


# Function to compute the multiplicative inverse of an encrypted value using TenSEAL
# Works best on the interval [0.5, 1]
# Source: https://en.wikipedia.org/wiki/Division_algorithm#Newton%E2%80%93Raphson_division
# TODO: Add scaling factor to improve accuracy
def compute_multiplicative_inverse(context: ts.Context, encrypted_value: ts.CKKSTensor, P=32):
    # Start with an initial guess (encoded as a scalar)
    # For multiplicative inverse, good initial guess can be crucial - let's assume approx. inverse is known
    # This should be based on some estimation method
    inverse = ts.ckks_tensor(
        context, torch.ones(
            encrypted_value.shape
        ).mul(48 / 17).tolist()
    ).sub(
        encrypted_value.mul(32 / 17)
    )

    # Number of iterations required to achieve desired precision
    iterations = math.ceil(math.log2((P + 1) / math.log2(17)))

    # Newton-Raphson iteration to refine the inverse
    for _ in range(iterations):
        prod = encrypted_value.mul(inverse)  # d * x_n

        correction = ts.ckks_tensor(
            context, torch.ones(
                encrypted_value.shape
            ).mul(2).tolist()
        ).sub(
            prod
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

    # Example: Encrypted value
    encrypted_value = ts.ckks_tensor(
        context, [1, 0.8, 0.5]
    )
    inverse_encrypted = compute_multiplicative_inverse(
        context, encrypted_value
    )

    # Decrypt to verify
    result = inverse_encrypted.decrypt().tolist()
    print("Decoded result:", result)
