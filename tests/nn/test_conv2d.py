from torchseal.nn import Conv2d as EncryptedConv2d
from torch.nn import Conv2d as PlainConv2d

import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_conv2d():
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
    out_channels = 1
    in_channels = 1
    kernel_height = 7
    kernel_width = 7
    stride = 3
    padding = 0

    # Declare input dimensions
    batch_size = 1
    input_height = 28
    input_width = 28

    # Count the output dimensions
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1

    # Create weight and bias
    kernel = torch.randn(
        out_channels,
        in_channels,
        kernel_height,
        kernel_width,
        requires_grad=True
    )
    bias = torch.randn(out_channels, requires_grad=True)

    # Create the input tensor
    input_tensor = torch.randn(
        batch_size, in_channels, input_height, input_width, requires_grad=True
    )

    # Encrypt the input tensor
    enc_input_tensor = torchseal.ckks_wrapper(
        context, input_tensor.view(batch_size, -1)
    )

    # Create the plaintext convolution layer
    conv2d = PlainConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_height, kernel_width),
        stride=stride,
        padding=padding,
        bias=True,
    )
    conv2d.weight = torch.nn.Parameter(kernel)  # Set the weight
    conv2d.bias = torch.nn.Parameter(bias)  # Set the bias

    # Create the encrypted convolution layer
    enc_conv2d = EncryptedConv2d(
        in_channel=in_channels,
        out_channel=out_channels,
        kernel_size=(kernel_height, kernel_width),
        input_size=torch.Size(
            [batch_size, in_channels, input_height, input_width]
        ),
        stride=stride,
        padding=padding,
        weight=kernel,
        bias=bias,
    )

    # Calculate the output
    output = conv2d.forward(input_tensor)
    enc_output = enc_conv2d.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.do_decryption()
    dec_output_resized = dec_output.view(
        batch_size, out_channels, output_height, output_width
    )

    # Check the correctness of the convolution (with a tolerance of 5e-2)
    assert torch.allclose(
        dec_output_resized,
        output,
        atol=5e-2,
        rtol=0
    ), "Convolution layer failed!"

# TODO: Add gradient test
