from torchseal.nn import Conv2d as EncryptedConv2d
from torchseal.utils import approximate_toeplitz_multiple_channels, precise_toeplitz_multiple_channels
from torch.nn import Conv2d as PlainConv2d

import typing
import torch
import numpy as np
import random
import tenseal as ts
import torchseal
from torchseal.wrapper.ckks import CKKSWrapper


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
    # NOTE: High number of padding will cause the test to fail (due to the added noise by generating near-zero values)
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
        batch_size, in_channels, input_height, input_width, requires_grad=True
    )

    # Encrypt the input tensor
    enc_input_tensor = torchseal.ckks_wrapper(
        input_tensor.view(batch_size, -1), do_encryption=True
    )
    enc_input_tensor.requires_grad_(True)

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
        weight=torchseal.ckks_wrapper(
            approximate_toeplitz_multiple_channels(
                kernel,
                (in_channels, input_height, input_width),
                stride=stride,
                padding=padding
            ),
            do_encryption=True
        ),
        bias=torchseal.ckks_wrapper(
            torch.repeat_interleave(
                bias,
                output_height * output_width
            ),
            do_encryption=True
        ),
    )

    # Set both layer on training mode
    conv2d.train()
    enc_conv2d.train()

    # Calculate the output
    output = conv2d.forward(input_tensor)
    enc_output = enc_conv2d.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.decrypt().plaintext_data.clone()

    # Resized the output
    output_resized = output.view(
        batch_size, -1
    )

    # Check the correctness of the convolution (with a tolerance of 5e-2)
    assert torch.allclose(
        dec_output,
        output_resized,
        atol=5e-2,
        rtol=0
    ), "Convolution layer failed!"

    # Create random grad_output
    grad_output = torch.randn_like(output)

    # Encrypt the grad_output
    enc_grad_output = torchseal.ckks_wrapper(
        grad_output.clone().view(batch_size, -1), do_encryption=True
    )

    # Do backward pass
    output.backward(grad_output)
    enc_output.backward(enc_grad_output)

    # Check the correctness of input gradients (with a tolerance of 5e-2)
    assert enc_input_tensor.grad is not None and input_tensor.grad is not None, "Input gradients are None!"

    conv2d_input_grad_expanded = input_tensor.grad.view(
        batch_size, -1
    )

    enc_conv2d_input_grad = typing.cast(
        CKKSWrapper,
        enc_input_tensor.grad
    ).decrypt().plaintext_data.clone()

    assert torch.allclose(
        enc_conv2d_input_grad,
        conv2d_input_grad_expanded,
        atol=5e-2,
        rtol=0
    ), "Input gradient failed!"

    # Check the correctness of weight gradients (with a tolerance of 5e-2)
    assert enc_conv2d.weight.grad is not None and conv2d.weight.grad is not None, "Weight gradients are None!"

    conv2d_weight_grad_expanded = precise_toeplitz_multiple_channels(
        conv2d.weight.grad,
        (in_channels, input_height, input_width),
        stride=stride,
        padding=padding
    )

    enc_conv2d_weight_grad = typing.cast(
        CKKSWrapper,
        enc_conv2d.weight.grad
    ).decrypt().plaintext_data.clone()

    assert torch.allclose(
        enc_conv2d_weight_grad, conv2d_weight_grad_expanded, atol=5e-2, rtol=0
    ), "Weight gradient failed!"

    # Check the correctness of bias gradients (with a tolerance of 5e-2)
    assert enc_conv2d.bias.grad is not None and conv2d.bias.grad is not None, "Bias gradients are None!"

    conv2d_bias_grad_expanded = torch.repeat_interleave(
        conv2d.bias.grad, output_height * output_width
    )

    enc_conv2d_bias_grad = typing.cast(
        CKKSWrapper,
        enc_conv2d.bias.grad
    ).decrypt().plaintext_data.clone()

    assert torch.allclose(
        enc_conv2d_bias_grad, conv2d_bias_grad_expanded, atol=5e-2, rtol=0
    ), "Bias gradient failed!"


def test_conv2d_eval():
    # TODO: Add evaluation test

    pass
