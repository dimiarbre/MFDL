from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Simple MLP for regression
class HousingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


def learning_step(
    models: list,
    optimizers: list,
    data_iters: list,
    device: torch.device = torch.device("cpu"),
):
    # Local update
    for node, model in enumerate(models):
        optimizer = optimizers[node]
        try:
            batch = next(data_iters[node])
        except StopIteration:
            continue  # skip if no more data
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x).squeeze()
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
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
    trainloaders: list[data.DataLoader],
    testloader: data.DataLoader,
    input_dim: int,
    device: torch.device = torch.device("cpu"),
):
    num_nodes = graph.number_of_nodes()
    models = [HousingMLP(input_dim).to(device) for _ in range(num_nodes)]
    optimizers = [optim.Adam(models[i].parameters(), lr=1e-2) for i in range(num_nodes)]
    data_iters = [iter(trainloaders[i]) for i in range(num_nodes)]

    test_losses: list[list[float]] = [[] for _ in range(num_nodes)]

    for step in range(num_steps):
        print(f"Step {step + 1:>{len(str(num_steps))}}/{num_steps}", end="\r")
        learning_step(models, optimizers, data_iters, device)
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
    return mse_list


def main():
    # G: nx.Graph = nx.cycle_graph(20)
    G = nx.empty_graph(
        20
    )  # Identity graph: each node only connected to itself (no communication)

    G.add_edges_from(
        [(i, i) for i in range(20)]
    )  # Always keep this to make a useful graph
    input_dim = 8  # California housing dataset
    num_steps = 100

    n = G.number_of_nodes()
    trainloaders, testloader = load_housing(n)

    models, train_losses = run_decentralized_training(
        G,
        num_steps=num_steps,
        trainloaders=trainloaders,
        testloader=testloader,
        input_dim=input_dim,
        device=torch.device("cuda"),
    )

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
