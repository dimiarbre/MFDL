import torch
import torch.nn as nn
import torch.utils.data as data
from data_utils import split_data
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Simple MLP for regression
class HousingMLP(nn.Module):
    def __init__(self, input_dim, seed=421):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


def housing_model_initializer(seed=421):
    return HousingMLP(8, seed=seed)


def load_housing(
    total_nodes, test_fraction=0.2, nb_batches=16, test_batch_size=4096, seed=421
):
    g = torch.Generator()
    g.manual_seed(seed)
    dataset = fetch_california_housing()  # 20,640 samples
    X = torch.tensor(StandardScaler().fit_transform(dataset.data), dtype=torch.float32)
    y = torch.tensor(dataset.target, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed
    )

    train_dataloaders = split_data(
        X=X_train,
        y=y_train,
        total_nodes=total_nodes,
        nb_batches=nb_batches,
        generator=g,
    )

    test_dataloader = data.DataLoader(
        data.TensorDataset(X_test, y_test),
        batch_size=test_batch_size,
        shuffle=True,
        generator=g,
        # pin_memory=True,
        # num_workers=4,
    )

    return train_dataloaders, test_dataloader
