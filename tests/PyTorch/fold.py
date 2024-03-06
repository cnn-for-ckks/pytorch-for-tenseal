from torch.nn import Fold, Unfold
import torch

if __name__ == "__main__":
    unfold = Unfold(kernel_size=(3, 3), stride=1)
    fold = Fold(output_size=(28, 28), kernel_size=(3, 3), stride=1)

    image = torch.rand(1, 1, 28, 28)

    print("image.shape:", image.shape, end="\n\n")

    unfolded_image = unfold.forward(image)
    print("unfolded_image.shape:", unfolded_image.shape, end="\n\n")

    dec_image = fold.forward(
        unfolded_image
    )
    print("dec_image.shape:", dec_image.shape, end="\n\n")

    # Check if the original image and the folded image are the same
    print(
        "Are the original image and the folded image the same?",
        torch.allclose(image, dec_image)
    )
