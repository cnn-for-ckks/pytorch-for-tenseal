import tenseal as ts
import torch
import time

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

    # Set the auto rescale
    context.auto_rescale = True

    # Galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    # Declare parameters
    out_features = 1
    in_features = 9

    # Create weight and input tensor
    weight = torch.randn(out_features, in_features)
    input_tensor = torch.randn(in_features)

    # Create encrypted tensor
    enc_input_tensor = ts.ckks_vector(context, input_tensor.tolist())

    # Multiply the encrypted tensor with the weight tensor
    start_time = time.perf_counter()
    enc_output = enc_input_tensor.matmul(weight.t().tolist())
    end_time = time.perf_counter()

    # Decrypt the output tensor
    dec_output = enc_output.decrypt()
    dec_output_tensor = torch.tensor(dec_output)

    # Compare the output with the target
    target = input_tensor.matmul(weight.t())

    # Check the correctness of the convolution via the toeplitz matrix
    enc_error = (dec_output_tensor - target).abs().sum()
    sum_target = target.abs().sum()
    percent_of_enc_error = enc_error / sum_target * 100

    print(
        f"Time taken for encrypted multiplication (without batch): {end_time - start_time:.6f} seconds"
    )
    print(
        f"Percentage of encrypted convolution error (without batch): {percent_of_enc_error:.6f}%"
    )
