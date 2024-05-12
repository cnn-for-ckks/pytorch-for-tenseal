from torchseal.nn import Conv2d as EncryptedConv2d
from torchseal.utils import approximate_toeplitz_multiple_channels, precise_toeplitz_multiple_channels
from torchseal.optim import SGD as EncryptedSGD

from torch.nn import Conv2d as PlainConv2d
from torch.optim import SGD as PlainSGD

import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_conv2d_train():
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
    out_channels = 2
    in_channels = 2
    kernel_height = 3
    kernel_width = 3
    stride = 1
    padding = 1

    # Declare input dimensions
    batch_size = 1
    input_height = 4
    input_width = 4

    # Declare the training parameters
    lr = 0.1

    # Adjust for padding
    padded_input_height = input_height + 2 * padding
    padded_input_width = input_width + 2 * padding

    # Count the output dimensions
    output_height = (padded_input_height - kernel_height) // stride + 1
    output_width = (padded_input_width - kernel_width) // stride + 1

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
        batch_size, in_channels, input_height, input_width
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
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_height, kernel_width),
        input_size=(input_height, input_width),
        stride=stride,
        padding=padding,
        weight=torch.nn.Parameter(
            approximate_toeplitz_multiple_channels(
                kernel,
                (in_channels, input_height, input_width),
                stride=stride,
                padding=padding
            )
        ),
        bias=torch.nn.Parameter(
            torch.repeat_interleave(
                bias, output_height * output_width
            )
        ),
    )

    # Set both layer on training mode
    conv2d.train()
    enc_conv2d.train()

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

    # Define the optimizer
    optim = PlainSGD(conv2d.parameters(), lr=lr)
    enc_optim = EncryptedSGD(enc_conv2d.parameters(), lr=lr)

    # Clear the gradients
    optim.zero_grad()
    enc_optim.zero_grad()

    # Create random grad_output
    grad_output = torch.randn_like(output)
    enc_grad_output = grad_output.view(batch_size, -1)

    # Do backward pass
    output.backward(grad_output)
    enc_output.backward(enc_grad_output)

    # TODO: Check the correctness of input gradients

    # Do the optimization step
    optim.step()
    enc_optim.step()

    # Check the correctness of parameters optimization (with a tolerance of 5e-2)
    conv2d_weight_expanded = precise_toeplitz_multiple_channels(
        conv2d.weight,
        (in_channels, input_height, input_width),
        stride=stride,
        padding=padding
    )

    assert torch.allclose(
        enc_conv2d.weight, conv2d_weight_expanded, atol=5e-2, rtol=0
    ), "Weight optimization failed!"

    conv2d_bias_expanded = conv2d.bias.repeat_interleave(
        output_height * output_width
    )

    assert torch.allclose(
        enc_conv2d.bias, conv2d_bias_expanded, atol=5e-2, rtol=0
    ), "Bias optimization failed!"


def test_conv2d_eval():
    # TODO: Add evaluation test

    pass
