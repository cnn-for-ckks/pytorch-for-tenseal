from torchseal.wrapper import CKKSWrapper
from torchseal.nn import ReLU as EncryptedReLU
from torch.nn import ReLU as PlainReLU

import typing
import torch
import numpy as np
import random
import tenseal as ts
import torchseal


# NOTE: We cannot use a high degree polynomial approximation because of the innacuracy when using a lot of multiplications in CKKS
def test_relu():
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
    start = -5
    stop = 5
    num_of_sample = 100
    degree = 6
    approximation_type = "minimax"
    deriv_start = -5
    deriv_stop = 5
    deriv_num_of_sample = 100
    deriv_degree = 6
    deriv_approximation_type = "minimax"

    # Declare input dimensions
    input_length = 20
    batch_size = 1

    # Create the input tensor
    input_tensor = torch.randn(batch_size, input_length, requires_grad=True)

    # Encrypt the input tensor
    enc_input_tensor = torchseal.ckks_wrapper(
        input_tensor, do_encryption=True
    )
    enc_input_tensor.requires_grad_(True)

    # Create the plaintext ReLU layer
    relu = PlainReLU()

    # Create the encrypted ReLU layer
    enc_relu = EncryptedReLU(
        start=start,
        stop=stop,
        num_of_sample=num_of_sample,
        degree=degree,
        approximation_type=approximation_type,
        deriv_start=deriv_start,
        deriv_stop=deriv_stop,
        deriv_num_of_sample=deriv_num_of_sample,
        deriv_degree=deriv_degree,
        deriv_approximation_type=deriv_approximation_type
    )

    # Calculate the output
    output = relu.forward(input_tensor)
    enc_output = enc_relu.forward(enc_input_tensor)

    # Decrypt the output
    dec_output = enc_output.decrypt().plaintext_data.clone()

    # Check the correctness of the convolution (with a tolerance of 0.25)
    assert torch.allclose(
        output, dec_output, atol=0.25, rtol=0
    ), "ReLU layer failed!"

    # Create random grad_output
    grad_output = torch.randn_like(output)

    # Encrypt the grad_output
    enc_grad_output = torchseal.ckks_wrapper(
        grad_output.clone(), do_encryption=True
    )

    # Do backward pass
    output.backward(grad_output)
    enc_output.backward(enc_grad_output)

    # Check the correctness of input gradients (with a tolerance of 0.75, this is due to the inaccuracy of approximating non-continuous functions with polynomials)
    assert enc_input_tensor.grad is not None and input_tensor.grad is not None, "Input gradients are None!"

    # Decrypt the input gradients
    dec_input_grad = typing.cast(
        CKKSWrapper,
        enc_input_tensor.grad
    ).decrypt().plaintext_data.clone()

    assert torch.allclose(
        dec_input_grad,
        input_tensor.grad,
        atol=0.75,
        rtol=0
    ), "Input gradients are incorrect!"
