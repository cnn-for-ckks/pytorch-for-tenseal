from torchseal.nn import Conv2d as EncryptedConv2d
from torchseal.utils import approximate_toeplitz_multiple_channels, precise_toeplitz_multiple_channels
from torch.nn import Conv2d as PlainConv2d

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

    # Set the context
    torchseal.set_context(context)

    # Declare parameters
    out_channels = 2
    in_channels = 2
    kernel_height = 3
    kernel_width = 3
    stride = 1
    padding = 0

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
        input_tensor.view(batch_size, -1), do_encryption=True
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
    dec_output = enc_output.decrypt()
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

    # Create random grad_output
    grad_output = torch.randn_like(output)
    enc_grad_output = grad_output.view(batch_size, -1)

    # Do backward pass
    output.backward(grad_output)
    enc_output.backward(enc_grad_output)

    # TODO: Check the correctness of input gradients

    # Check the correctness of weight gradients (with a tolerance of 5e-2)
    assert enc_conv2d.weight.grad is not None and conv2d.weight.grad is not None, "Weight gradients are None!"

    conv2d_weight_grad_expanded = precise_toeplitz_multiple_channels(
        conv2d.weight.grad,
        (in_channels, input_height, input_width),
        stride=stride,
        padding=padding
    )

    assert torch.allclose(
        enc_conv2d.weight.grad, conv2d_weight_grad_expanded, atol=5e-2, rtol=0
    ), "Weight gradient failed!"

    # Check the correctness of bias gradients (with a tolerance of 5e-2)
    assert enc_conv2d.bias.grad is not None and conv2d.bias.grad is not None, "Bias gradients are None!"

    conv2d_bias_grad_expanded = torch.repeat_interleave(
        conv2d.bias.grad, output_height * output_width
    )

    assert torch.allclose(
        enc_conv2d.bias.grad, conv2d_bias_grad_expanded, atol=5e-2, rtol=0
    ), "Bias gradient failed!"


def test_conv2d_eval():
    # TODO: Add evaluation test

    pass
