from torch.nn.functional import unfold, fold

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
        [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] for _ in range(8)]]
    )

    # NOTE: Plaintext im2col
    image = torch.tensor(
        [[
            [
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
            ] for _ in range(28)
        ]]
    )
    print("image.shape:", image.shape, end="\n\n")
    print(image, end="\n\n")

    unfolded_image = unfold(image, kernel_size=(8, 8), stride=3)
    print("unfolded_image.shape:", unfolded_image.shape, end="\n\n")

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
    num_row = unfolded_image.shape[0]
    num_col = unfolded_image.shape[1]

    # NOTE: Encrypted convolution
    # Create the convolutional weight
    enc_conv_weight = ts.plain_tensor(weight.view(-1))

    # Perform the convolution
    enc_result = enc_unfolded_image.enc_matmul_plain(
        enc_conv_weight.tolist(), num_col
    )

    # Decrypt the result
    print("enc_result:", enc_result.decrypt(), end="\n\n")

    # NOTE: Plaintext col2im (Currently only works when kernel size is power of 2)
    # NOTE: Also still broken
    raw_dec_unfolded_image = enc_unfolded_image.decrypt()
    dec_unfolded_image = torch.tensor(raw_dec_unfolded_image).reshape(
        len(raw_dec_unfolded_image) // num_col, num_col
    )
    print("dec_unfolded_image.shape", dec_unfolded_image.shape, end="\n\n")

    dec_image = fold(
        dec_unfolded_image, output_size=(28, 28), kernel_size=(8, 8), stride=3
    )
    print("dec_image.shape:", dec_image.shape, end="\n\n")
    print(dec_image, end="\n\n")
