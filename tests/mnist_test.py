from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
from mnist_train import ConvNet


import numpy as np
import torch


def test(model: ConvNet, test_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))

    # model in evaluation mode
    model.eval()

    for data, target in test_loader:
        output = model.forward(data)
        loss = criterion.forward(output, target)
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader)
    print(f"Test Loss: {test_loss:.6f}\n")

    for label in range(10):
        print(
            f"Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% "
            f"({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})"
        )

    print(
        f"\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% "
        f"({int(np.sum(class_correct))}/{int(np.sum(class_total))})"
    )


if __name__ == "__main__":
    # Set the seed for reproducibility
    torch.manual_seed(73)

    # Load the data
    test_data = datasets.MNIST(
        "data", train=False, download=True, transform=transforms.ToTensor()
    )

    # Set the batch size
    batch_size = 64

    # Create the samplers
    sampler = RandomSampler(test_data, num_samples=50, replacement=True)

    # Create the data loaders
    test_loader = DataLoader(
        test_data, batch_size=batch_size, sampler=sampler
    )

    # Load the model
    model = ConvNet()
    model.load_state_dict(torch.load("./parameters/MNIST/model.pth"))

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Test the model
    test(model, test_loader, criterion)
