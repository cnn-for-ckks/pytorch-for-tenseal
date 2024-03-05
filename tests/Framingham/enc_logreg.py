from typing import Optional
from torch.nn import Module
from torch.utils.data import DataLoader
from torchseal.function import SigmoidFunction
from torchseal.wrapper.ckks import CKKSWrapper
from logreg import LogisticRegression

import torch
import tenseal as ts
import torchseal


class EncLogisticRegression(Module):
    def __init__(self, n_features: int, torch_nn: Optional[LogisticRegression] = None) -> None:
        super(EncLogisticRegression, self).__init__()

        self.linear = torchseal.nn.Linear(
            n_features,
            1,
            torch_nn.linear.weight.data,
            torch_nn.linear.bias.data
        ) if torch_nn is not None else torchseal.nn.Linear(n_features, 1)

    def forward(self, x: CKKSWrapper) -> CKKSWrapper:
        # Fully connected layer
        first_result = self.linear.forward(x)

        # Sigmoid activation function
        first_result_activated: CKKSWrapper = SigmoidFunction.apply(
            first_result
        )  # type: ignore

        return first_result_activated


def enc_train(context: ts.Context, enc_model: EncLogisticRegression, train_loader: DataLoader, criterion: torch.nn.BCELoss, optimizer: torch.optim.SGD, n_epochs: int = 10) -> EncLogisticRegression:
    # Model in training mode
    enc_model.train()

    for epoch in range(n_epochs):
        train_loss = 0.

        for raw_data, raw_target in train_loader:
            optimizer.zero_grad()

            # Encrypt the data
            data = raw_data[0].tolist()
            enc_data = ts.ckks_vector(context, data)
            enc_data_wrapper = CKKSWrapper(torch.rand(len(data)), enc_data)

            # Encrypted evaluation
            enc_output = enc_model.forward(enc_data_wrapper)

            # Decryption of result
            output = enc_output.do_decryption().clamp(0, 1)

            # Compute loss
            target = raw_target[0]
            loss = criterion.forward(output, target)
            loss.backward()
            optimizer.step()  # BUG: Weight and bias are not accurately updated
            train_loss += loss.item()

        # Calculate average losses
        train_loss = 0 if len(
            train_loader
        ) == 0 else train_loss / len(train_loader)

        print(
            "Epoch: {} \tTraining Loss (Ciphertext): {:.6f}".format(
                epoch + 1, train_loss
            )
        )

    # Model in evaluation mode
    enc_model.eval()

    return enc_model


def enc_test(context: ts.Context, enc_model: EncLogisticRegression, test_loader: DataLoader, criterion: torch.nn.BCELoss) -> None:
    # Initialize lists to monitor test loss and accuracy
    test_loss = 0.

    # Model in evaluation mode
    enc_model.eval()

    for raw_data, raw_target in test_loader:
        # Encryption
        data = raw_data[0].tolist()
        enc_data = ts.ckks_vector(context, data)
        enc_data_wrapper = CKKSWrapper(torch.rand(len(data)), enc_data)

        # Encrypted evaluation
        enc_output = enc_model.forward(enc_data_wrapper)

        # Decryption using client secret key
        output = enc_output.do_decryption().clamp(0, 1)

        # Compute loss
        target = raw_target[0]
        loss = criterion.forward(output, target)
        test_loss += loss.item()

        print(f"Current Test Loss (Ciphertext): {loss.item():.6f}")

    # Calculate and print avg test loss
    test_loss = 0 if len(test_loader) == 0 else test_loss / len(test_loader)
    print(f"\nAverage Test Loss (Ciphertext): {test_loss:.6f}")
