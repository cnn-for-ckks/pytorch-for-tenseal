from torch.nn.functional import conv2d
from scipy.linalg import toeplitz

import numpy as np
import torch


def toeplitz_1_ch(kernel, input_size):
    # shapes
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    o_h = i_h-k_h+1

    # construct 1d conv toeplitz matrices for each row of the kernel
    results = []
    for r in range(k_h):
        results.append(
            toeplitz(
                c=(kernel[r, 0], *np.zeros(i_w-k_w)), r=(*kernel[r], *np.zeros(i_w-k_w))
            )
        )

    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = results[0].shape

    W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block))

    for i, B in enumerate(results):
        for j in range(o_h):
            W_conv[j, :, i+j, :] = B

    W_conv.shape = (h_blocks*h_block, w_blocks*w_block)

    return W_conv


def toeplitz_mult_ch(kernel, input_size):
    """Compute toeplitz matrix for 2d conv with multiple in and out channels.
    Args:
        kernel: shape=(n_out, n_in, H_k, W_k)
        input_size: (n_in, H_i, W_i)"""

    kernel_size = kernel.shape
    output_size = (kernel_size[0], input_size[1] -
                   (kernel_size[1]-1), input_size[2] - (kernel_size[2]-1))
    T = np.zeros((output_size[0], int(
        np.prod(output_size[1:])), input_size[0], int(np.prod(input_size[1:]))))

    for i, ks in enumerate(kernel):  # loop over output channel
        for j, k in enumerate(ks):  # loop over input channel
            T_k = toeplitz_1_ch(k, input_size[1:])
            T[i, :, j, :] = T_k

    T.shape = (np.prod(output_size), np.prod(input_size))

    return T


if __name__ == "__main__":
    k = np.random.randn(4*3*3*3).reshape((4, 3, 3, 3))
    i = np.random.randn(3, 7, 9)

    T = toeplitz_mult_ch(k, i.shape)
    out = T.dot(i.flatten()).reshape((1, 4, 5, 7))

    # check correctness of convolution via toeplitz matrix
    print(np.sum(
        (out - conv2d(torch.tensor(i).view(1, 3, 7, 9), torch.tensor(k)).numpy())**2))
