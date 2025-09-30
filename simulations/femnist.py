import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from data_utils import split_data
from torchvision import datasets, transforms


# Simple CNN for FEMNIST (28x28 grayscale images, 62 classes)
class FEMNISTCNN(nn.Module):
    def __init__(self, seed=421):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 62),  # FEMNIST: 62 classes
        )

    def forward(self, x):
        return self.net(x)


def femnist_model_initializer(seed=421):
    return FEMNISTCNN(seed)


def dirichlet_partition(X, y, total_nodes, alpha=0.5, batch_size=32, generator=None):
    """
    Splits dataset (X, y) into total_nodes partitions using Dirichlet distribution.

    Args:
        X (Tensor): Features tensor [N, ...].
        y (Tensor): Labels tensor [N].
        total_nodes (int): Number of nodes/partitions.
        alpha (float): Dirichlet concentration parameter.
        batch_size (int): Batch size for each DataLoader.
        generator (torch.Generator, optional): For reproducibility.

    Returns:
        List[DataLoader]: List of DataLoaders, one per node.
    """
    num_classes = int(y.max().item()) + 1
    idxs = [np.where(y.numpy() == i)[0] for i in range(num_classes)]
    proportions = np.random.dirichlet([alpha] * total_nodes, num_classes)

    node_indices = [[] for _ in range(total_nodes)]
    for c, idx_c in enumerate(idxs):
        np.random.shuffle(idx_c)
        splits = (proportions[c] * len(idx_c)).astype(int)
        # Adjust splits to sum to len(idx_c)
        splits[-1] = len(idx_c) - splits[:-1].sum()
        start = 0
        for node, count in enumerate(splits):
            node_indices[node].extend(idx_c[start : start + count])
            start += count

    # Shuffle indices for each node
    for node in range(total_nodes):
        np.random.shuffle(node_indices[node])

    dataloaders = []
    for node in range(total_nodes):
        X_node = X[node_indices[node]]
        y_node = y[node_indices[node]]
        ds = data.TensorDataset(X_node, y_node)
        dataloaders.append(
            data.DataLoader(
                ds, batch_size=batch_size, shuffle=True, generator=generator
            )
        )
    return dataloaders


def load_femnist(
    total_nodes,
    test_fraction=0.2,
    train_batch_size=32,
    test_batch_size=4096,
    seed=421,
    data_dir="./data",
):
    g = torch.Generator()
    g.manual_seed(seed)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Standard MNIST normalization
        ]
    )
    # Download FEMNIST as EMNIST 'byclass' split
    train_dataset = datasets.EMNIST(
        root=data_dir, split="byclass", train=True, download=True, transform=transform
    )
    test_dataset = datasets.EMNIST(
        root=data_dir, split="byclass", train=False, download=True, transform=transform
    )

    # Convert to tensors for splitting
    X_train = train_dataset.data.unsqueeze(1).float() / 255.0  # [N, 1, 28, 28]
    y_train = train_dataset.targets
    X_test = test_dataset.data.unsqueeze(1).float() / 255.0
    y_test = test_dataset.targets

    # Optionally shuffle before splitting
    perm = torch.randperm(len(X_train), generator=g)
    X_train, y_train = X_train[perm], y_train[perm]

    train_dataloaders = split_data(
        X=X_train,
        y=y_train,
        total_nodes=total_nodes,
        batch_size=train_batch_size,
        generator=g,
    )

    test_dataloader = data.DataLoader(
        data.TensorDataset(X_test, y_test),
        batch_size=test_batch_size,
        shuffle=True,
        generator=g,
    )

    return train_dataloaders, test_dataloader


def main():
    # Parameters
    total_nodes = 5
    train_batch_size = 32
    test_batch_size = 256
    seed = 42

    # Load data
    train_dataloaders, test_dataloader = load_femnist(
        total_nodes=total_nodes,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        seed=seed,
    )

    # Print info about the dataloaders
    print(f"Number of train dataloaders (nodes): {len(train_dataloaders)}")
    for i, dl in enumerate(train_dataloaders):
        print(f"Node {i}: {len(dl.dataset)} samples, {len(dl)} batches")

    print(
        f"Test set: {len(test_dataloader.dataset)} samples, {len(test_dataloader)} batches"
    )

    total_train_samples = sum(len(dl.dataset) for dl in train_dataloaders)
    print(f"Total train samples: {total_train_samples}")

    # Instantiate model and print summary
    model = FEMNISTCNN(seed=seed)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Use Dirichlet partitioner instead of default split_data
    X_train = train_dataloaders[0].dataset.tensors[0].new_empty(0)
    y_train = train_dataloaders[0].dataset.tensors[1].new_empty(0, dtype=torch.long)
    for dl in train_dataloaders:
        X_train = torch.cat([X_train, dl.dataset.tensors[0]], dim=0)
        y_train = torch.cat([y_train, dl.dataset.tensors[1]], dim=0)

    dirichlet_dataloaders = dirichlet_partition(
        X=X_train,
        y=y_train,
        total_nodes=total_nodes,
        alpha=0.5,
        batch_size=train_batch_size,
        generator=torch.Generator().manual_seed(seed),
    )

    print("\n[Dirichlet Partition]")
    for i, dl in enumerate(dirichlet_dataloaders):
        print(f"Node {i}: {len(dl.dataset)} samples, {len(dl)} batches")
        labels = dl.dataset.tensors[1].numpy()
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = dict(zip(unique, counts))
        print(f"  Class distribution: {class_counts}")


if __name__ == "__main__":
    main()
