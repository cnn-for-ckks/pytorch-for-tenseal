from torch import Tensor
from torch.utils.data import Dataset

import pandas as pd
import torch


class FraminghamDataset(Dataset):
    def __init__(self, csv_file: str) -> None:
        # Load data
        data = pd.read_csv(csv_file)

        # Drop rows with missing values
        data = data.dropna()

        # Drop some features
        data = data.drop(
            columns=[
                "education",
                "currentSmoker",
                "BPMeds",
                "diabetes",
                "diaBP",
                "BMI"
            ]
        )

        # Balance data
        grouped = data.groupby("TenYearCHD")
        data = grouped.apply(
            lambda x: x.sample(
                grouped.size().min(),
                random_state=73
            ).reset_index(drop=True)
        )

        # Extract target
        self.target = torch.tensor(
            data["TenYearCHD"].values
        ).float().unsqueeze(1)

        # Drop target
        data = data.drop(columns=["TenYearCHD"])

        # Normalize data
        data = (data - data.mean()) / data.std()

        # Extract features
        self.features = torch.tensor(data.values).float()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.features[idx], self.target[idx]
