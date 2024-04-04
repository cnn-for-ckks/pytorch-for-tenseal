from torch.nn.functional import unfold

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

    # Declare global parameters
    output_size = (9, 9)
    kernel_size = (3, 3)
    stride = 3
    padding = 0

    # Randomize the tensor
    weight = torch.rand(1, 1, 3, 3)
    image = torch.rand(1, 9, 9)

    # Plaintext im2col
    unfolded_image = unfold(
        image,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # Plaintext convolution
    # Create the convolutional weight
    conv_weight = weight.view(1, -1)

    # Perform the convolution
    conv_output = conv_weight.matmul(
        unfolded_image
    ).view(-1)

    # Encrypted im2col
    enc_unfolded_image = ts.enc_matmul_encoding(
        context, unfolded_image.t()
    )
    num_row = unfolded_image.shape[0]
    num_col = unfolded_image.shape[1]

    # Encrypted convolution
    # Create the convolutional weight
    enc_conv_weight = weight.view(-1)

    # Perform the convolution
    enc_result = enc_unfolded_image.enc_matmul_plain(
        enc_conv_weight.tolist(), num_col
    )

    # Plaintext col2im
    result = torch.tensor(enc_result.decrypt()).view(1, 1, 3, 3)

    # Compare the output with the target
    output = weight.view(1, -1).matmul(
        unfolded_image
    ).view(1, 1, 3, 3)
    target = torch.nn.functional.conv2d(
        image.view(1, 1, 9, 9), weight, stride=3, padding=0
    )

    # Check the correctness of the convolution via the im2col encoding
    print("Plaintext convolution:", (output - target).abs().sum())
    print("Encrypted convolution:", (result - target).abs().sum())
