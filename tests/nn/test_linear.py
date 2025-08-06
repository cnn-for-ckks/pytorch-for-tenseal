from torchseal.wrapper.ckks import CKKSWrapper
from torchseal.nn import Linear as EncryptedLinear
from torch.nn import Linear as PlainLinear

import typing
import torch
import numpy as np
import random
import tenseal as ts
import torchseal


def test_linear_train():
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
    out_features = 8
    in_features = 8

    # Declare input dimensions
    batch_size = 1

    # Create weight and bias
    weight = torch.randn(out_features, in_features, requires_grad=True)
    bias = torch.randn(out_features, requires_grad=True)

    print()
    print("Weight:", weight)
    print("Bias:", bias)

    # Create the input tensor
    input_tensor = torch.randn(batch_size, in_features, requires_grad=True)

    print()
    print("Input tensor:", input_tensor)

    # Encrypt the input tensor
    enc_input_tensor = torchseal.ckks_wrapper(
        input_tensor.clone().view(batch_size, -1), do_encryption=True
    )
    enc_input_tensor.requires_grad_(True)

    # Create the plaintext linear layer
    linear = PlainLinear(in_features, out_features)
    linear.weight = torch.nn.Parameter(weight)
    linear.bias = torch.nn.Parameter(bias)

    # Create the encrypted linear layer
    enc_linear = EncryptedLinear(
        in_features=in_features,
        out_features=out_features,
        weight=torchseal.ckks_wrapper(
            weight.clone(), do_encryption=True
        ),
        bias=torchseal.ckks_wrapper(
            bias.clone(), do_encryption=True
        )
    )

    # Set both layer on training mode
    linear.train()
    enc_linear.train()

    # Calculate the output
    output = linear.forward(input_tensor)
    enc_output = enc_linear.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.decrypt().plaintext_data.clone()

    print()
    print("Output:", output)
    print("Decrypted output:", dec_output)
    print(f"Max difference: {torch.max(torch.abs(output - dec_output)).item():.4f}")
    print(f"Min difference: {torch.min(torch.abs(output - dec_output)).item():.4f}")
    print(f"Avg difference: {torch.mean(torch.abs(output - dec_output)).item():.4f}")

    # Check the correctness of the convolution (with a tolerance of 5e-2)
    assert torch.allclose(
        output, dec_output, atol=5e-2, rtol=0
    ), "Linear layer failed!"

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

    dec_linear_input_grad = typing.cast(
        CKKSWrapper,
        enc_input_tensor.grad
    ).decrypt().plaintext_data.clone()

    print()
    print("Encrypted input gradient:", dec_linear_input_grad)
    print("Plaintext input gradient:", input_tensor.grad)
    print(f"Max difference: {torch.max(torch.abs(dec_linear_input_grad - input_tensor.grad)).item():.4f}")
    print(f"Min difference: {torch.min(torch.abs(dec_linear_input_grad - input_tensor.grad)).item():.4f}")
    print(f"Avg difference: {torch.mean(torch.abs(dec_linear_input_grad - input_tensor.grad)).item():.4f}")

    assert torch.allclose(
        dec_linear_input_grad,
        input_tensor.grad,
        atol=5e-2,
        rtol=0
    ), "Input gradient failed!"

    # Check the correctness of weight gradients (with a tolerance of 5e-2)
    assert enc_linear.weight.grad is not None and linear.weight.grad is not None, "Weight gradients are None!"

    dec_linear_weight_grad = typing.cast(
        CKKSWrapper,
        enc_linear.weight.grad
    ).decrypt().plaintext_data.clone()

    print()
    print("Encrypted weight gradient:", dec_linear_weight_grad)
    print("Plaintext weight gradient:", linear.weight.grad)
    print(f"Max difference: {torch.max(torch.abs(dec_linear_weight_grad - linear.weight.grad)).item():.4f}")
    print(f"Min difference: {torch.min(torch.abs(dec_linear_weight_grad - linear.weight.grad)).item():.4f}")
    print(f"Avg difference: {torch.mean(torch.abs(dec_linear_weight_grad - linear.weight.grad)).item():.4f}")

    assert torch.allclose(
        dec_linear_weight_grad,
        linear.weight.grad,
        atol=5e-2,
        rtol=0
    ), "Weight gradient failed!"

    # Check the correctness of bias gradients (with a tolerance of 5e-2)
    assert enc_linear.bias.grad is not None and linear.bias.grad is not None, "Bias gradients are None!"

    dec_linear_bias_grad = typing.cast(
        CKKSWrapper,
        enc_linear.bias.grad
    ).decrypt().plaintext_data.clone()

    print()
    print("Encrypted bias gradient:", dec_linear_bias_grad)
    print("Plaintext bias gradient:", linear.bias.grad)
    print(f"Max difference: {torch.max(torch.abs(dec_linear_bias_grad - linear.bias.grad)).item():.4f}")
    print(f"Min difference: {torch.min(torch.abs(dec_linear_bias_grad - linear.bias.grad)).item():.4f}")
    print(f"Avg difference: {torch.mean(torch.abs(dec_linear_bias_grad - linear.bias.grad)).item():.4f}")

    assert torch.allclose(
        dec_linear_bias_grad,
        linear.bias.grad,
        atol=5e-2,
        rtol=0
    ), "Bias gradient failed!"


def test_linear_eval():
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
    out_features = 8
    in_features = 8

    # Declare input dimensions
    batch_size = 1

    # Create weight and bias
    weight = torch.randn(out_features, in_features, requires_grad=True)
    bias = torch.randn(out_features, requires_grad=True)

    # Create the input tensor
    input_tensor = torch.randn(batch_size, in_features, requires_grad=True)

    # Encrypt the input tensor
    enc_input_tensor = torchseal.ckks_wrapper(
        input_tensor.clone().view(batch_size, -1), do_encryption=True
    )
    enc_input_tensor.requires_grad_(True)

    # Create the plaintext linear layer
    linear = PlainLinear(in_features, out_features)
    linear.weight = torch.nn.Parameter(weight)
    linear.bias = torch.nn.Parameter(bias)

    # Create the encrypted linear layer
    enc_linear = EncryptedLinear(
        in_features=in_features,
        out_features=out_features,
        weight=torchseal.ckks_wrapper(
            weight.clone(), do_encryption=True
        ),
        bias=torchseal.ckks_wrapper(
            bias.clone(), do_encryption=True
        )
    )

    # Set both layer on eval mode
    linear.eval()
    enc_linear.eval()

    # Calculate the output
    output = linear.forward(input_tensor)
    enc_output = enc_linear.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.decrypt().plaintext_data.clone()

    # Check the correctness of the convolution (with a tolerance of 5e-2)
    assert torch.allclose(
        output, dec_output, atol=5e-2, rtol=0
    ), "Linear layer failed!"
