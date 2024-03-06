from torch.nn import Unfold

import torch

if __name__ == "__main__":
    unfold = Unfold(kernel_size=(5, 5), stride=1)

    image = torch.rand(1, 1, 28, 28)

    output = unfold.forward(image)

    print("Output shape:", output.size())
