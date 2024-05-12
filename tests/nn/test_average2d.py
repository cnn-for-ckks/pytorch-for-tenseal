from torchseal.nn import AvgPool2d as EncryptedAvgPool2d
from torch.nn import AvgPool2d as PlainAvgPool2d

import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_avgpool2d():
    # Set the seed for reproducibility
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)

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
    n_channels = 2
    kernel_height = 3
    kernel_width = 3
    stride = 1
    padding = 1

    # Declare input dimensions
    batch_size = 1
    input_height = 4
    input_width = 4

    # Adjust for padding
    padded_input_height = input_height + 2 * padding
    padded_input_width = input_width + 2 * padding

    # Count the output dimensions
    output_height = (padded_input_height - kernel_height) // stride + 1
    output_width = (padded_input_width - kernel_width) // stride + 1

    # Create the input tensor
    input_tensor = torch.randn(
        batch_size, n_channels, input_height, input_width, requires_grad=True
    )

    # Encrypt the input tensor
    enc_input_tensor = torchseal.ckks_wrapper(
        context, input_tensor.view(batch_size, -1)
    )

    # Create the plaintext average pooling layer
    plain_avgpool2d = PlainAvgPool2d(
        kernel_size=(kernel_height, kernel_width),
        stride=stride,
        padding=padding
    )

    # Create the encrypted average pooling layer
    enc_avgpool2d = EncryptedAvgPool2d(
        n_channels=n_channels,
        input_size=(input_height, input_width),
        kernel_size=(kernel_height, kernel_width),
        stride=stride,
        padding=padding
    )

    # Calculate the output
    output = plain_avgpool2d.forward(input_tensor)
    enc_output = enc_avgpool2d.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.do_decryption()
    dec_output_resized = dec_output.view(
        batch_size, n_channels, output_height, output_width
    )

    # Compare the results (with a tolerance of 5e-2)
    assert torch.allclose(
        dec_output_resized,
        output,
        atol=5e-2,
        rtol=0
    ), "Average pooling layer failed!"

    # TODO: Do backward pass and check the correctness of the input gradients
