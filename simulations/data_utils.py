import math

import torch
import torch.utils.data as data


def split_data(
    X, y, total_nodes: int, nb_batches, generator: torch.Generator
) -> list[data.DataLoader]:
    idx = torch.arange(len(X))

    dataloaders = []
    for node_id in range(total_nodes):
        # Simple partition: split by node_id
        node_idx = idx[node_id::total_nodes]
        ds = data.TensorDataset(X[node_idx], y[node_idx])
        batch_size = max(1, math.ceil(len(X) / nb_batches))
        dataloaders.append(
            data.DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            )
        )
    # Ensure all dataloaders have the same number of batches
    min_len = min(len(dl) for dl in dataloaders)
    for i, dl in enumerate(dataloaders):
        if len(dl) > min_len:
            # Remove extra data so that this dataloader has min_len batches
            ds = dl.dataset
            batch_size = dl.batch_size
            keep_n = min_len * batch_size
            dataloaders[i] = data.DataLoader(
                data.TensorDataset(ds.tensors[0][:keep_n], ds.tensors[1][:keep_n]),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                # pin_memory=True,
                # num_workers=4,
            )
    return dataloaders
