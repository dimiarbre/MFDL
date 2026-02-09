import math

import numpy as np
import torch
import torch.utils.data as data


def make_batch_sampler_indices(num_samples, nb_batches, rng):
    batch_sizes = [num_samples // nb_batches] * nb_batches
    for i in range(num_samples % nb_batches):
        batch_sizes[i] += 1
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    batch_indices = []
    start = 0
    for size in batch_sizes:
        batch_indices.append(indices[start : start + size].tolist())
        start += size
    return batch_indices


def split_data(X, y, total_nodes: int, nb_batches, rng) -> list[data.DataLoader]:
    idx = np.arange(len(X))
    rng.shuffle(idx)

    dataloaders = []
    for node_id in range(total_nodes):
        # Simple partition: split by node_id
        node_idx = idx[node_id::total_nodes]
        ds = data.TensorDataset(X[node_idx], y[node_idx])

        batch_indices = make_batch_sampler_indices(
            len(ds), nb_batches=nb_batches, rng=rng
        )

        dataloaders.append(
            data.DataLoader(
                ds,
                batch_sampler=batch_indices,
            )
        )

    for node_id, dataloader in enumerate(dataloaders):
        assert (
            len(dataloader) == nb_batches
        ), f"Number of batches generated {len(dataloader)} does not match expected number of batches {nb_batches} for node {node_id}"
    return dataloaders
