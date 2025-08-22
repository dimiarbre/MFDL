from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Simple MLP for regression
class HousingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


def decentralized_step(
    models: list,
    optimizers: list,
    data_iters: list,
    graph: nx.Graph,
    device: torch.device = torch.device("cpu"),
):
    # Local update
    for node in graph.nodes:
        model = models[node]
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

    # Model averaging with neighbors
    with torch.no_grad():
        # Save current parameters for all models
        current_states = [
            {k: v.clone() for k, v in model.state_dict().items()} for model in models
        ]
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node)) + [node]
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
    data_partitioner: Callable[[int], data.DataLoader],
    input_dim: int,
    device: torch.device = torch.device("cpu"),
):
    num_nodes = graph.number_of_nodes()
    models = [HousingMLP(input_dim).to(device) for _ in range(num_nodes)]
    optimizers = [optim.Adam(models[i].parameters(), lr=1e-2) for i in range(num_nodes)]
    dataloaders = [data_partitioner(i) for i in range(num_nodes)]
    data_iters = [iter(dataloaders[i]) for i in range(num_nodes)]

    test_losses: list[list[float]] = [[] for _ in range(num_nodes)]

    for step in range(num_steps):
        print(f"Step {step + 1:>{len(str(num_steps))}}/{num_steps}", end="\r")
        decentralized_step(models, optimizers, data_iters, graph, device)
        step_losses = evaluate_models(models, data_partitioner, device)
        for node_id, loss in enumerate(step_losses):
            test_losses[node_id].append(loss)

    return models, test_losses


# Example data partitioner (to be replaced with your own)
def example_data_partitioner(node_id: int) -> data.DataLoader:
    # Replace this with your own partitioning logic
    dataset = fetch_california_housing()
    X = torch.tensor(StandardScaler().fit_transform(dataset.data), dtype=torch.float32)
    y = torch.tensor(dataset.target, dtype=torch.float32)
    # Simple partition: split by node_id
    idx = torch.arange(len(X))
    node_idx = idx[node_id::5]  # assuming 5 nodes for example
    ds = data.TensorDataset(X[node_idx], y[node_idx])
    return data.DataLoader(ds, batch_size=32, shuffle=True)


def evaluate_models(models: list, data_partitioner, device):
    mse_list = []
    for node_id, model in enumerate(models):
        model.eval()
        loader = data_partitioner(node_id)
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                output = model(x).squeeze()
                preds.append(output.cpu())
                targets.append(y.cpu())
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        mse = mean_squared_error(targets, preds)
        mse_list.append(mse)
    return mse_list


if __name__ == "__main__":
    G = nx.cycle_graph(20)
    input_dim = 8  # California housing dataset
    num_steps = 100
    models, train_losses = run_decentralized_training(
        G,
        num_steps=num_steps,
        data_partitioner=example_data_partitioner,
        input_dim=input_dim,
        device=torch.device("cuda"),
    )

    plt.figure()
    for node_id in range(len(models)):
        plt.plot(range(num_steps), train_losses[node_id])

    plt.title("Train losses")

    plt.figure()
    mse_list = evaluate_models(
        models, example_data_partitioner, device=torch.device("cuda")
    )
    plt.bar(range(len(mse_list)), mse_list)
    plt.xlabel("Node")
    plt.ylabel("MSE")
    plt.title("Model MSE per Node")
    plt.show()
