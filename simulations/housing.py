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
from mfdl_optimizer import MFDLSGD
from opacus import GradSampleModule
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Remove warnings for Housing that should be safe, raised because of Opacus + Housing
warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients.",
)


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
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_microbatch_grad()
            pred = model(x).squeeze()
            loss = nn.functional.mse_loss(pred, y)
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
    nb_micro_batches: int,
    trainloaders: list[data.DataLoader],
    testloader: data.DataLoader,
    input_dim: int,
    device: torch.device = torch.device("cpu"),
):
    num_nodes = graph.number_of_nodes()
    models = [
        GradSampleModule(HousingMLP(input_dim)).to(device) for _ in range(num_nodes)
    ]
    # optimizers = [optim.Adam(models[i].parameters(), lr=1e-2) for i in range(num_nodes)]
    optimizers = [
        MFDLSGD(models[i].parameters(), C=C, C_sens=1, lr=1e-2, device=device)
        for i in range(num_nodes)
    ]
    data_iters = [iter(trainloaders[i]) for i in range(num_nodes)]

    test_losses: list[list[float]] = [[] for _ in range(num_nodes)]

    for step in range(num_steps):
        print(f"Step {step + 1:>{len(str(num_steps))}}/{num_steps}", end="\r")
        learning_step(
            models,
            nb_micro_batches=nb_micro_batches,
            optimizers=optimizers,
            data_iters=data_iters,
            dataloaders=trainloaders,
            device=device,
        )
        average_step(graph, models)
        step_losses = evaluate_models(models, testloader, device)
        for node_id, loss in enumerate(step_losses):
            test_losses[node_id].append(loss)

    return models, test_losses


def split_data(X, y, total_nodes: int, batch_size) -> list[data.DataLoader]:
    idx = torch.arange(len(X))

    dataloaders = []
    for node_id in range(total_nodes):
        # Simple partition: split by node_id
        node_idx = idx[node_id::total_nodes]
        ds = data.TensorDataset(X[node_idx], y[node_idx])
        dataloaders.append(data.DataLoader(ds, batch_size=batch_size, shuffle=True))
    return dataloaders


def load_housing(total_nodes, test_fraction=0.2, batch_size=32):
    dataset = fetch_california_housing()
    X = torch.tensor(StandardScaler().fit_transform(dataset.data), dtype=torch.float32)
    y = torch.tensor(dataset.target, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction)
    train_dataloaders = split_data(
        X=X_train, y=y_train, total_nodes=total_nodes, batch_size=batch_size
    )

    test_dataloader = data.DataLoader(
        data.TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True
    )

    return train_dataloaders, test_dataloader


def evaluate_models(models: list, dataloader, device):
    mse_list = []
    n = len(models)
    for node_id, model in enumerate(models):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
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


def main():
    G: nx.Graph

    # G: nx.Graph = nx.cycle_graph(5)
    # G: nx.Graph = nx.cycle_graph(20)

    G: nx.Graph = nx.empty_graph(5)
    # G: nx.Graph = nx.empty_graph(
    #     20
    # )  # Identity graph: each node only connected to itself (no communication)

    # G: nx.Graph = nx.cycle_graph(100)

    G.add_edges_from(
        [(i, i) for i in range(G.number_of_nodes())]
    )  # Always keep this to make a useful graph
    input_dim = 8  # California housing dataset
    num_steps = 100
    micro_batches_size = 100
    nb_micro_batches = 5

    n = G.number_of_nodes()
    trainloaders, testloader = load_housing(n, batch_size=micro_batches_size)
    print([len(loader) for loader in trainloaders])

    C_LDP = np.identity(num_steps)

    start_time = time.time()
    models, train_losses = run_decentralized_training(
        G,
        num_steps=num_steps,
        C=C_LDP,
        nb_micro_batches=nb_micro_batches,
        trainloaders=trainloaders,
        testloader=testloader,
        input_dim=input_dim,
        device=torch.device("cuda"),
    )
    elapsed_time = time.time() - start_time
    print(f"Training took {elapsed_time:.2f} seconds.")

    plt.figure()
    for node_id in range(len(models)):
        plt.plot(range(num_steps), train_losses[node_id])

    plt.title("Train losses")

    plt.figure()
    mse_list = evaluate_models(models, testloader, device=torch.device("cuda"))
    plt.bar(range(len(mse_list)), mse_list)
    plt.xlabel("Node")
    plt.ylabel("MSE")
    plt.title("Model MSE per Node")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(421)
    # np.random.seed(421)
    # random.seed(421)

    main()
