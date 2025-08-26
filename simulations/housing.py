import time
import warnings
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import workloads_generator
from mfdl_optimizer import MFDLSGD
from opacus import GradSampleModule
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from workloads_generator import compute_sensitivity

# Remove warnings for Housing that should be safe, raised because of Opacus + Housing
warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients.",
)

NB_RESETS = 0


# Simple MLP for regression
class HousingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


def learning_step(
    models: list,
    nb_micro_batches: int,
    optimizers: list,
    data_iters: list,
    dataloaders: list,
    device: torch.device = torch.device("cpu"),
):
    global NB_RESETS
    # Local update
    for node, model in enumerate(models):
        optimizer: MFDLSGD = optimizers[node]
        for _ in range(nb_micro_batches):
            try:
                batch = next(data_iters[node])
            except StopIteration:
                data_iters[node] = iter(
                    dataloaders[node]
                )  # Reset the current dataloader.
                batch = next(data_iters[node])
                if node == 0:
                    NB_RESETS += 1
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_microbatch_grad()
            pred = model(x)

            # Cast into 1D tensor even when the batch is of size 0.
            pred = pred.view(-1)
            loss = nn.functional.mse_loss(pred, y.view(-1))

            loss.backward()
            optimizer.microbatch_step()
        optimizer.step()


def average_step(graph, models):
    """
    Runs a step of gossip averaging along graph on models
    """
    # Model averaging with neighbors
    with torch.no_grad():
        # Save current parameters for all models
        current_states = [
            {k: v.clone() for k, v in model.state_dict().items()} for model in models
        ]
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            if node not in neighbors:
                neighbors.append(node)
            assert node in neighbors  # Make sure we keep the local model.

            # Average parameters using saved states
            avg_state = {}
            for key in current_states[node]:
                avg_state[key] = sum(current_states[n][key] for n in neighbors) / len(
                    neighbors
                )
            models[node].load_state_dict(avg_state)


def run_decentralized_training(
    graph: nx.Graph,
    num_steps: int,
    C: np.ndarray,
    micro_batches_per_epoch: int,
    trainloaders: list[data.DataLoader],
    testloader: data.DataLoader,
    input_dim: int,
    device: torch.device = torch.device("cpu"),
):
    global NB_RESETS
    NB_RESETS = 0
    num_nodes = graph.number_of_nodes()
    models = [
        GradSampleModule(HousingMLP(input_dim)).to(device) for _ in range(num_nodes)
    ]
    participations_intervals = [
        len(trainloader) // micro_batches_per_epoch for trainloader in trainloaders
    ]
    # optimizers = [optim.Adam(models[i].parameters(), lr=1e-2) for i in range(num_nodes)]
    optimizers = [
        MFDLSGD(
            models[i].parameters(),
            C=C,
            participation_interval=participations_intervals[i],
            lr=1e-2,
            device=device,
            id=i,
        )
        for i in range(num_nodes)
    ]
    data_iters = [iter(trainloaders[i]) for i in range(num_nodes)]

    test_losses: list[list[float]] = [[] for _ in range(num_nodes)]

    for step in range(num_steps):
        print(f"Step {step + 1:>{len(str(num_steps))}}/{num_steps}", end="\r")
        learning_step(
            models,
            nb_micro_batches=micro_batches_per_epoch,
            optimizers=optimizers,
            data_iters=data_iters,
            dataloaders=trainloaders,
            device=device,
        )
        average_step(graph, models)
        step_losses = evaluate_models(models, testloader, device)
        for node_id, loss in enumerate(step_losses):
            test_losses[node_id].append(loss)

    # Test losses of shape [nb_steps, nb_nodes]. Time is the 1st axis, nodes the second.
    res_test_losses = np.array(test_losses).T
    print(f"Simulation needed {NB_RESETS} passes through the dataset")
    return models, res_test_losses


def split_data(X, y, total_nodes: int, batch_size) -> list[data.DataLoader]:
    idx = torch.arange(len(X))

    dataloaders = []
    for node_id in range(total_nodes):
        # Simple partition: split by node_id
        node_idx = idx[node_id::total_nodes]
        ds = data.TensorDataset(X[node_idx], y[node_idx])
        dataloaders.append(data.DataLoader(ds, batch_size=batch_size, shuffle=True))
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
                pin_memory=True,
                # num_workers=4,
            )
    return dataloaders


def load_housing(
    total_nodes, test_fraction=0.2, train_batch_size=32, test_batch_size=4096
):
    dataset = fetch_california_housing()
    X = torch.tensor(StandardScaler().fit_transform(dataset.data), dtype=torch.float32)
    y = torch.tensor(dataset.target, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction)
    train_dataloaders = split_data(
        X=X_train, y=y_train, total_nodes=total_nodes, batch_size=train_batch_size
    )

    test_dataloader = data.DataLoader(
        data.TensorDataset(X_test, y_test),
        batch_size=test_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    return train_dataloaders, test_dataloader


def evaluate_models(models: list, dataloader, device):
    mse_list = []
    n = len(models)
    with torch.no_grad():
        for node_id, model in enumerate(models):
            model.eval()
            preds, targets = [], []
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                output = model(x).squeeze()
                preds.append(output.cpu())
                targets.append(y.cpu())
            preds = torch.cat(preds).numpy()
            targets = torch.cat(targets).numpy()
            mse = mean_squared_error(targets, preds)
            mse_list.append(mse)
            model.train()
    return mse_list


def expander_graph(n, d):
    if d < n:
        G = nx.random_regular_graph(d, n)
    else:
        raise ValueError(
            "Degree d must be less than number of nodes n for a regular graph."
        )
    return G


def main():
    G: nx.Graph

    device = torch.device("cuda")
    print(f"Device : {device}")

    n = 100

    # G: nx.Graph = nx.empty_graph(n)
    # G: nx.Graph = nx.cycle_graph(n)
    # G = expander_graph(n, np.ceil(np.log(n)))
    G = nx.complete_graph(n)
    # G = nx.watts_strogatz_graph(n, k=10, p=0.3)

    G.add_edges_from(
        [(i, i) for i in range(G.number_of_nodes())]
    )  # Always keep this to make a useful graph
    input_dim = 8  # California housing dataset
    num_repetition = 20  # Nb of full pass over the data
    micro_batches_size = 50
    micro_batches_per_epoch = 1

    n = G.number_of_nodes()
    trainloaders, testloader = load_housing(n, train_batch_size=micro_batches_size)
    nb_micro_batches = [len(loader) for loader in trainloaders]
    assert np.all(np.array(nb_micro_batches) == np.min(nb_micro_batches))
    nb_micro_batches = np.min(nb_micro_batches)

    num_steps = num_repetition * (nb_micro_batches // micro_batches_per_epoch)

    # Here, we only consider "local" correlations, hence nb_nodes = 1.
    C_NONOISE = np.zeros((num_steps, num_steps))
    C_LDP = workloads_generator.MF_LDP(nb_nodes=1, nb_iterations=num_steps)
    C_ANTIPGD = workloads_generator.MF_ANTIPGD(nb_nodes=1, nb_iterations=num_steps)

    all_test_losses = {}
    plt.figure()
    for name, C in [
        ("Unnoised baseline", C_NONOISE),
        ("LDP", C_LDP),
        ("ANTIPGD", C_ANTIPGD),
    ]:
        print(f"Running simulation for {name}")
        sens = compute_sensitivity(
            C,
            participation_interval=nb_micro_batches // micro_batches_per_epoch,
            num_epochs=num_steps,
        )
        print(f"sens²(C_{name}) = {sens**2}")

        start_time = time.time()
        _, all_test_losses[name] = run_decentralized_training(
            G,
            num_steps=num_steps,
            C=C,
            micro_batches_per_epoch=micro_batches_per_epoch,
            trainloaders=trainloaders,
            testloader=testloader,
            input_dim=input_dim,
            device=device,
        )
        elapsed_time = time.time() - start_time
        print(f"Training took {elapsed_time:.2f} seconds.")

    for name, test_losses in all_test_losses.items():
        # Plot the min and max

        print(test_losses.shape)
        avg_loss = test_losses.mean(axis=1)
        min_loss = test_losses.min(axis=1)
        max_loss = test_losses.max(axis=1)

        (line,) = plt.plot(range(num_steps), avg_loss, label=name)
        color = line.get_color()
        plt.fill_between(range(num_steps), min_loss, max_loss, alpha=0.2, color=color)

    plt.legend()
    plt.grid()

    plt.title("Test losses per model")
    plt.xlabel("Communication rounds")
    plt.ylabel("Test loss")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(421)
    np.random.seed(421)
    # random.seed(421)

    main()
