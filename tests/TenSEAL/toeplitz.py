from torchseal.utils import toeplitz_multiple_channels

import tenseal as ts
import torch

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

    # Declare parameters
    kernel = torch.randn(1, 1, 7, 7)
    input_tensor = torch.randn(1, 28, 28)

    # Encrypt the input tensor
    enc_input_tensor = ts.ckks_vector(context, input_tensor.view(-1).tolist())

    # Create the toeplitz matrix
    toeplitz_matrix = toeplitz_multiple_channels(
        kernel, input_tensor.shape, stride=3, padding=0
    )

    # Multiply the toeplitz matrix with the encrypted input tensor
    enc_output = enc_input_tensor.matmul(toeplitz_matrix.t().tolist())

    # Decrypt the output tensor
    dec_output = enc_output.decrypt()
    dec_output_tensor = torch.tensor(dec_output).view(1, 1, 8, 8)

    # Compare the output with the target
    output_tensor = toeplitz_matrix.matmul(
        input_tensor.view(-1)
    ).view(1, 1, 8, 8)
    target = torch.nn.functional.conv2d(
        input_tensor.view(1, 1, 28, 28), kernel, stride=3, padding=0
    )

    # Check the correctness of the convolution via the toeplitz matrix
    print("Plaintext convolution:", (output_tensor - target).abs().sum())
    print("Encrypted convolution:", (dec_output_tensor - target).abs().sum())
