import tenseal as ts

if __name__ == "__main__":
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

    # Create encrypted tensor
    enc_input_tensor = ts.ckks_tensor(context, [[1, 2], [3, 4]])

    # Create weight tensor
    weight = [[1, 0], [0, 1]]

    # Multiply the encrypted tensor with the weight tensor
    enc_output = enc_input_tensor.mm(weight)

    # Decrypt the output tensor
    dec_output = enc_output.decrypt()

    # Print the output tensor
    print(dec_output.tolist())
