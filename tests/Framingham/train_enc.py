from typing import Optional
from torch.nn import Module
from tenseal import CKKSVector
from torchseal.function import SigmoidFunction
from train import LogisticRegression

import torchseal


class EncLogisticRegression(Module):
    def __init__(self, n_features: int, torch_nn: Optional[LogisticRegression]) -> None:
        super(EncLogisticRegression, self).__init__()

        self.linear = torchseal.nn.Linear(
            n_features,
            1,
            torch_nn.linear.weight.T.data,
            torch_nn.linear.bias.data
        ) if torch_nn is not None else torchseal.nn.Linear(n_features, 1)

    def forward(self, x: CKKSVector) -> CKKSVector:
        # Fully connected layer
        first_result = self.linear.forward(x)

        # Sigmoid activation function
        first_result_activated = SigmoidFunction.apply(first_result)

        return first_result_activated
