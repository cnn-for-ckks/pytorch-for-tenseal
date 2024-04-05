from torchseal.utils import toeplitz_multiple_channels

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
    out_channels = 1
    in_channels = 1
    kernel_height = 7
    kernel_width = 7
    stride = 3
    padding = 0
    batch_size = 1
    input_height = 28
    input_width = 28

    # Count the output dimensions
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1

    # Create weight and input tensor
    kernel = torch.randn(
        out_channels,
        in_channels,
        kernel_height,
        kernel_width
    )
    input_tensor = torch.randn(
        batch_size, in_channels, input_height, input_width
    )

    # Encrypt the input tensor
    enc_input_tensor = ts.ckks_tensor(
        context, input_tensor.view(batch_size, -1).tolist()
    )

    # Create the toeplitz matrix
    toeplitz_matrix = toeplitz_multiple_channels(
        kernel, input_tensor.shape[1:], stride=stride, padding=padding
    )

    # Multiply the toeplitz matrix with the encrypted input tensor
    start_time = time.perf_counter()
    enc_output = enc_input_tensor.mm(toeplitz_matrix.t().tolist())
    end_time = time.perf_counter()

    # Decrypt the output tensor
    dec_output = enc_output.decrypt()
    dec_output_tensor = torch.tensor(dec_output.tolist()).view(
        batch_size, out_channels, output_height, output_width
    )

    # Compare the output with the target
    output_tensor = input_tensor.view(batch_size, -1).matmul(
        toeplitz_matrix.t()
    ).view(batch_size, out_channels, output_height, output_width)

    target = torch.nn.functional.conv2d(
        input_tensor.view(batch_size, in_channels, input_height, input_width), kernel, stride=stride, padding=padding
    )

    # Check the correctness of the convolution via the toeplitz matrix
    error = (output_tensor - target).abs().sum()
    enc_error = (dec_output_tensor - target).abs().sum()
    sum_target = target.abs().sum()
    percent_of_error = error / sum_target * 100
    percent_of_enc_error = enc_error / sum_target * 100

    print(
        f"Time taken for encrypted multiplication (with batch): {end_time - start_time:.6f} seconds"
    )
    print(
        f"Percentage of plaintext convolution error (with batch): {percent_of_error:.6f}%"
    )
    print(
        f"Percentage of encrypted convolution error (with batch): {percent_of_enc_error:.6f}%"
    )
