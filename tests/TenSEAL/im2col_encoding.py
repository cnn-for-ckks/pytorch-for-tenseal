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

    # Randomize the tensor
    weight = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] for _ in range(7)]]
    )

    # NOTE: Plaintext im2col
    image = torch.rand(1, 28, 28)
    image = torch.tensor(
        [[
            [
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
            ] for _ in range(28)
        ]]
    )
    unfolded_image = unfold(image, kernel_size=(7, 7), stride=3)

    # NOTE: Plaintext convolution
    # Create the convolutional weight
    conv_weight = weight.view(1, -1)

    # Perform the convolution
    conv_output = conv_weight.matmul(
        unfolded_image
    ).view(-1)

    # print the result
    print("conv_output:", conv_output.tolist(), end="\n\n")

    # NOTE: Encrypted im2col
    enc_unfolded_image = ts.enc_matmul_encoding(
        context, unfolded_image.t()
    )
    windows_nb = unfolded_image.shape[1]

    # NOTE: Encrypted convolution
    # Create the convolutional weight
    enc_conv_weight = ts.plain_tensor(weight.view(-1))

    # Perform the convolution
    enc_result = enc_unfolded_image.enc_matmul_plain(
        enc_conv_weight.tolist(), windows_nb
    )

    # Decrypt the result
    print("enc_result:", enc_result.decrypt(), end="\n\n")
