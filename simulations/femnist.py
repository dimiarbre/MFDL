import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from data_utils import make_batch_sampler_indices
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import GroupedNaturalIdPartitioner
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader
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


class FEMNIST_tiny_CNN(nn.Module):
    def __init__(self, seed=421):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            # nn.GroupNorm(4, 16),  # 4 groups for 16 channels → 4 channels per group
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.GroupNorm(8, 32),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 62),
            # nn.ReLU(),
            # nn.Linear(128, 62),  # FEMNIST: 62 classes
        )

    def forward(self, x):
        return self.net(x)


def femnist_model_initializer(seed=421):
    return FEMNIST_tiny_CNN(seed=seed)


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


def to_torch_data(partition):
    to_tensor = transforms.ToTensor()
    images = torch.stack([to_tensor(x) for x in partition["image"]])
    labels = torch.tensor(partition["character"])

    return data.TensorDataset(images, labels)


def load_femnist(
    total_nodes,
    test_fraction=0.2,
    nb_batches=16,
    test_batch_size=4096,
    seed=421,
    data_dir="./data",
):
    rng = np.random.default_rng(seed)
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={
            "train": GroupedNaturalIdPartitioner(
                partition_by="writer_id", group_size=10
            )
        },
        seed=seed,
    )
    train_dataloaders = []
    test_datasets = []
    for i in range(total_nodes):
        partition = fds.load_partition(partition_id=i)

        # Convert partition to torch dataset
        dataset = to_torch_data(partition)
        images = dataset.tensors[0]
        labels = dataset.tensors[1]

        train_imgs, test_imgs, train_labels, test_labels = train_test_split(
            images,
            labels,
            test_size=test_fraction,
            random_state=seed,  # , stratify=labels
        )

        batch_indices = make_batch_sampler_indices(len(train_imgs), nb_batches, rng=rng)
        train_dataset = data.TensorDataset(train_imgs, train_labels)
        # batch_sampler = data.BatchSampler(
        #     batch_indices, batch_size=None, drop_last=False
        # )
        torch_dataloader = data.DataLoader(train_dataset, batch_sampler=batch_indices)

        # train_batch_size = max(1, math.ceil(len(train_imgs) / nb_batches))

        # torch_dataloader = data.DataLoader(
        #     data.TensorDataset(train_imgs, train_labels),
        #     batch_size=train_batch_size,
        #     shuffle=True,
        #     drop_last=False,
        # )
        train_dataloaders.append(torch_dataloader)

        test_datasets.append(data.TensorDataset(test_imgs, test_labels))

    test_dataset = ConcatDataset(test_datasets)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    return train_dataloaders, test_dataloader


def main():
    # Parameters
    total_nodes = 15
    nb_batches = 16
    test_batch_size = 4096
    seed = 42
    lr = 0.5
    epochs = 100

    # Helper to extract per-node sample indices from train dataloaders
    def extract_node_indices(train_dataloaders):
        nodes_indices = []
        for dl in train_dataloaders:
            # each dl.dataset is a TensorDataset with tensors (images, labels)
            try:
                imgs = dl.dataset.tensors[0]
            except Exception:
                # If dataset is a Subset/ConcatDataset, try to access .dataset
                imgs = dl.dataset.tensors[0]
            # We don't have original global indices; construct a fingerprint using
            # the tensor contents' hashes (fast approximate) to compare equality.
            # For reproducibility check we compare shapes and a small hash.
            flat = imgs.reshape(len(imgs), -1)
            # compute a deterministic checksum per node
            checksum = (
                int(torch.sum(flat[0 : min(10, len(flat))]).item())
                if len(flat) > 0
                else 0
            )
            nodes_indices.append((len(imgs), checksum))
        return nodes_indices

    # Load data twice with the same seed and compare partitions
    train_dataloaders_a, test_dataloader = load_femnist(
        total_nodes=total_nodes,
        nb_batches=nb_batches,
        test_batch_size=test_batch_size,
        seed=seed,
    )

    train_dataloaders_b, _ = load_femnist(
        total_nodes=total_nodes,
        nb_batches=nb_batches,
        test_batch_size=test_batch_size,
        seed=seed,
    )

    nodes_a = extract_node_indices(train_dataloaders_a)
    nodes_b = extract_node_indices(train_dataloaders_b)

    print("Reproducibility check with same seed:")
    same = True
    for i, (a, b) in enumerate(zip(nodes_a, nodes_b)):
        print(
            f" Node {i}: runA -> samples={a[0]}, checksum={a[1]} | runB -> samples={b[0]}, checksum={b[1]}"
        )
        if a != b:
            same = False
    if same:
        print(
            "PASS: Two runs with the same seed produced identical partitions (by shape+checksum)."
        )
    else:
        print("FAIL: Partitions differ between two runs with the same seed.")

    # Also check that a different seed produces at least one different partition
    other_seed = seed + 1
    train_dataloaders_c, _ = load_femnist(
        total_nodes=total_nodes,
        nb_batches=nb_batches,
        test_batch_size=test_batch_size,
        seed=other_seed,
    )
    nodes_c = extract_node_indices(train_dataloaders_c)
    diff_found = any(a != c for a, c in zip(nodes_a, nodes_c))
    print(
        f"Different-seed check (seed={seed} vs seed={other_seed}): {'DIFFER' if diff_found else 'ALL SAME'}"
    )

    # Print info about the dataloaders
    print(f"Number of train dataloaders (nodes): {len(train_dataloaders_a)}")
    for i, dl in enumerate(train_dataloaders_a):
        print(f"Node {i}: {len(dl.dataset)} samples, {len(dl)} batches")

    print(
        f"Test set: {len(test_dataloader.dataset)} samples, {len(test_dataloader)} batches"
    )

    total_train_samples = sum(len(dl.dataset) for dl in train_dataloaders_a)
    print(f"Total train samples: {total_train_samples}")

    # Instantiate model and print summary
    model = femnist_model_initializer(seed=seed)
    # model = FEMNISTCNN(seed)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}. LR: {lr}")

    import matplotlib.pyplot as plt

    # Quick training loop to evaluate the model in a centralized setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Concatenate all train datasets and create a single shuffled DataLoader
    all_train_dataset = ConcatDataset([dl.dataset for dl in train_dataloaders_a])
    shuffled_train_dataloader = DataLoader(
        all_train_dataset, batch_size=2024, shuffle=True
    )

    train_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for X, y in shuffled_train_dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            _, preds = outputs.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        avg_loss = running_loss / total
        acc = correct / total
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc*100:.4f}%")

    # Plot training loss
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid()
    plt.title("FEMNIST Training Loss")

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_total = 0
    test_correct = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            test_loss += loss.item() * X.size(0)
            _, preds = outputs.max(1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)
    avg_test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    print(f"Test Loss: {avg_test_loss:.4f} - Test Acc: {test_acc:.4f}")

    plt.show()
    # # Use Dirichlet partitioner instead of default split_data
    # X_train = train_dataloaders[0].dataset.tensors[0].new_empty(0)
    # y_train = train_dataloaders[0].dataset.tensors[1].new_empty(0, dtype=torch.long)
    # for dl in train_dataloaders:
    #     X_train = torch.cat([X_train, dl.dataset.tensors[0]], dim=0)
    #     y_train = torch.cat([y_train, dl.dataset.tensors[1]], dim=0)

    # dirichlet_dataloaders = dirichlet_partition(
    #     X=X_train,
    #     y=y_train,
    #     total_nodes=total_nodes,
    #     alpha=0.5,
    #     batch_size=train_batch_size,
    #     generator=torch.Generator().manual_seed(seed),
    # )

    # print("\n[Dirichlet Partition]")
    # for i, dl in enumerate(dirichlet_dataloaders):
    #     print(f"Node {i}: {len(dl.dataset)} samples, {len(dl)} batches")
    #     labels = dl.dataset.tensors[1].numpy()
    #     unique, counts = np.unique(labels, return_counts=True)
    #     class_counts = dict(zip(unique, counts))
    #     print(f"  Class distribution: {class_counts}")


if __name__ == "__main__":
    main()
