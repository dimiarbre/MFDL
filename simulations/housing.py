import argparse
import concurrent.futures
import functools
import multiprocessing
import os
import time
import warnings
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotters
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
from utils import get_communication_matrix, get_graph
from workloads_generator import compute_sensitivity

# Remove warnings for Housing that should be safe, raised because of Opacus + Housing
warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients.",
)


# Simple MLP for regression
class HousingMLP(nn.Module):
    def __init__(self, input_dim, seed=421):
        super().__init__()
        torch.manual_seed(seed)
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
    nb_resets = 0
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
                    nb_resets += 1
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
    return nb_resets


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
    epsilon: float,
    micro_batches_per_epoch: int,
    trainloaders: list[data.DataLoader],
    testloader: data.DataLoader,
    input_dim: int,
    lr: float = 1e-1,
    device: torch.device = torch.device("cpu"),
):
    nb_resets = 0
    num_nodes = graph.number_of_nodes()
    # Ensure all models start from the same initial weights by seeding before each model creation
    models = []
    for _ in range(num_nodes):
        torch.manual_seed(421)
        np.random.seed(421)
        model = GradSampleModule(HousingMLP(input_dim)).to(device)
        models.append(model)
    participations_intervals = [
        len(trainloader) // micro_batches_per_epoch for trainloader in trainloaders
    ]
    # optimizers = [optim.Adam(models[i].parameters(), lr=1e-2) for i in range(num_nodes)]
    optimizers = []
    for i in range(num_nodes):
        batch_size = trainloaders[i].batch_size
        assert batch_size is not None, "Batch size should not be None"
        optimizers.append(
            MFDLSGD(
                models[i].parameters(),
                C=C,
                participation_interval=participations_intervals[i],
                lr=lr,
                device=device,
                id=i,
                noise_multiplier=1 / epsilon,
                batch_size=batch_size * micro_batches_per_epoch,
            )
        )
    data_iters = [iter(trainloaders[i]) for i in range(num_nodes)]

    test_losses: list[list[float]] = [[] for _ in range(num_nodes)]

    for step in range(num_steps):
        print(f"Step {step + 1:>{len(str(num_steps))}}/{num_steps}", end="\r")
        nb_resets += learning_step(
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
    print(f"Simulation needed {nb_resets} passes through the dataset")
    return models, res_test_losses


def split_data(
    X, y, total_nodes: int, batch_size, generator: torch.Generator
) -> list[data.DataLoader]:
    idx = torch.arange(len(X))

    dataloaders = []
    for node_id in range(total_nodes):
        # Simple partition: split by node_id
        node_idx = idx[node_id::total_nodes]
        ds = data.TensorDataset(X[node_idx], y[node_idx])
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


def load_housing(
    total_nodes, test_fraction=0.2, train_batch_size=32, test_batch_size=4096, seed=421
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
        batch_size=train_batch_size,
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


def run_simulation(
    args,
    G: nx.Graph,
    num_steps,
    micro_batches_per_epoch,
    epsilon: float,
    input_dim,
    device_name: str,
    nb_micro_batches,
    dataloader_seed,
    lr: float,
    micro_batches_size,
):
    np.random.seed(dataloader_seed)
    torch.manual_seed(dataloader_seed)
    device = torch.device(device_name)
    n = G.number_of_nodes()
    trainloaders, testloader = load_housing(
        n, train_batch_size=micro_batches_size, seed=dataloader_seed
    )
    name, C = args
    print(f"Running simulation for {name} (PID: {os.getpid()})")
    sens = compute_sensitivity(
        C,
        participation_interval=nb_micro_batches // micro_batches_per_epoch,
        nb_steps=num_steps,
    )
    print(f"sens²(C_{name}) = {sens**2}")

    start_time = time.time()
    _, test_losses = run_decentralized_training(
        G,
        num_steps=num_steps,
        C=C,
        epsilon=epsilon,
        micro_batches_per_epoch=micro_batches_per_epoch,
        trainloaders=trainloaders,
        testloader=testloader,
        input_dim=input_dim,
        lr=lr,
        device=device,
    )
    elapsed_time = time.time() - start_time
    print(f"{name}: Training took {elapsed_time:.2f} seconds.")
    return name, test_losses


def run_experiment(
    debug=True,
    use_optimals=False,
    nb_nodes=10,
    graph_name="expander",
    epsilon: float = 1.0,
    num_repetition=5,
    dataloader_seed=421,
    lr=1e-1,
    nb_micro_batches=16,
    recompute: bool = False,
):
    device_name = "cpu"
    device = torch.device(device_name)
    print(f"Device : {device}")

    # 20,640 total samples, 20,640 * 0.8 = 16,512 train samples
    micro_batches_size = 16512 // (nb_nodes * nb_micro_batches)
    micro_batches_per_epoch = 1

    input_dim = 8  # California housing dataset

    G = get_graph(graph_name, nb_nodes)
    nb_nodes = G.number_of_nodes()
    communication_matrix = get_communication_matrix(G)

    trainloaders, _ = load_housing(nb_nodes, train_batch_size=micro_batches_size)
    nb_micro_batches_list = [len(loader) for loader in trainloaders]
    print(nb_micro_batches_list)
    assert np.all(np.array(nb_micro_batches_list) == np.min(nb_micro_batches_list))
    nb_micro_batches_val = np.min(nb_micro_batches_list)

    num_steps = num_repetition * (nb_micro_batches_val // micro_batches_per_epoch)

    print(f"Number of steps per epoch: {num_steps // num_repetition}")

    # String of experiment details for plots
    current_experiment_properties = (
        f"graph{graph_name}_"
        f"nodes{nb_nodes}_"
        f"eps{epsilon}_"
        f"reps{num_repetition}_"
        f"mbpe{micro_batches_per_epoch}_"
        f"lr{lr}_"
        f"seed{dataloader_seed}_"
        f"mbsize{micro_batches_size}_"
        f"steps{num_steps}_"  # Should be redundant
    )
    details = (
        f"Graph: {graph_name} | "
        f"Nb nodes: {nb_nodes} | "
        f"Micro-batches per epoch: {micro_batches_per_epoch} | ",
        f"Nb micro batches: {nb_micro_batches} | "
        f"Micro-batch size: {micro_batches_size} | "
        f"Num_repetitions: {num_repetition}",
    )
    csv_path = f"results/housing/simulation_{current_experiment_properties}.csv"

    # Check if results already exist and skip if not recomputing
    if (not recompute) and os.path.exists(csv_path):
        print(f"Results already exist at {csv_path}. Skipping computation.")
        return

    # Here, we only consider "local" correlations, hence nb_nodes = 1.
    C_NONOISE = np.zeros((num_steps, num_steps))
    C_LDP = workloads_generator.MF_LDP(nb_nodes=1, nb_iterations=num_steps)
    C_ANTIPGD = workloads_generator.MF_ANTIPGD(nb_nodes=1, nb_iterations=num_steps)
    C_BSR_LOCAL = workloads_generator.BSR_local_factorization(nb_iterations=num_steps)

    all_test_losses = {}

    configs = [
        ("Unnoised baseline", C_NONOISE),
        ("LDP", C_LDP),
        ("ANTIPGD", C_ANTIPGD),
        ("BSR_LOCAL", C_BSR_LOCAL),
    ]

    if use_optimals:
        print("Starting computation of C_OPTIMAL_LOCAL...")
        start_time_local = time.time()
        _, C_OPTIMAL_LOCAL = workloads_generator.MF_OPTIMAL_local(
            communication_matrix=communication_matrix,
            nb_nodes=nb_nodes,
            nb_steps=num_steps,
            nb_epochs=num_repetition,
        )
        elapsed_time_local = time.time() - start_time_local
        print(
            f"Finished computation of C_OPTIMAL_LOCAL in {elapsed_time_local:.2f} seconds."
        )

        print("Starting computation of C_OPTIMAL_DL...")
        start_time_dl = time.time()
        _, C_OPTIMAL_DL = workloads_generator.MF_OPTIMAL_DL(
            communication_matrix=communication_matrix,
            nb_nodes=nb_nodes,
            nb_steps=num_steps,
            nb_epochs=num_repetition,
        )
        elapsed_time_dl = time.time() - start_time_dl
        print(f"Finished computation of C_OPTIMAL_DL in {elapsed_time_dl:.2f} seconds.")

        plotters.plot_factorization(
            C_OPTIMAL_DL,
            title="C Optimal DL workload",
            details=details,
            save_name_properties=f"DLopti_{current_experiment_properties}",
            debug=debug,
        )
        plotters.plot_factorization(
            C_OPTIMAL_LOCAL,
            title="C optimal Local workload",
            details=details,
            save_name_properties=f"LOCALopti_{current_experiment_properties}",
            debug=debug,
        )

        compute_sensitivity(
            C_OPTIMAL_LOCAL,
            participation_interval=num_steps // num_repetition,
            nb_steps=num_steps,
        )

        compute_sensitivity(
            C_OPTIMAL_DL,
            participation_interval=num_steps // num_repetition,
            nb_steps=num_steps,
        )
        configs.append(("OPTIMAL", C_OPTIMAL_DL))
        configs.append(("OPTIMAL_LOCAL", C_OPTIMAL_LOCAL))

    # Use ProcessPoolExecutor for true parallelism with PyTorch
    run_sim_args = {
        "G": G,
        "num_steps": num_steps,
        "micro_batches_per_epoch": micro_batches_per_epoch,
        "epsilon": epsilon,
        "input_dim": input_dim,
        "device_name": device_name,
        "nb_micro_batches": nb_micro_batches_val,
        "dataloader_seed": dataloader_seed,
        "lr": lr,
        "micro_batches_size": micro_batches_size,
    }
    run_simulation_partial = functools.partial(run_simulation, **run_sim_args)

    if debug:
        results = []
        for config in configs:
            results.append(run_simulation_partial(config))
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(mp_context=context) as executor:
            results = list(executor.map(run_simulation_partial, configs))

    for name, test_losses in results:
        all_test_losses[name] = test_losses

    # Organize results into a DataFrame
    records = []
    for name, test_losses in all_test_losses.items():
        for step in range(test_losses.shape[0]):
            for node in range(test_losses.shape[1]):
                records.append(
                    {
                        "method": name,
                        "step": step,
                        "node": node,
                        "test_loss": test_losses[step, node],
                    }
                )

    df = pd.DataFrame(records)
    # Add experiment parameters to the DataFrame for each record
    df["graph_name"] = graph_name
    df["num_passes"] = num_repetition
    df["total_steps"] = num_steps
    df["num_nodes"] = nb_nodes
    df["num_repetitions"] = num_repetition
    df["micro_batch_size"] = micro_batches_size
    df["micro_batches_per_epoch"] = micro_batches_per_epoch
    df["nb_micro_batches"] = nb_micro_batches_val
    df["input_dim"] = input_dim
    df["device"] = device_name
    df["dataloader_seed"] = run_sim_args["dataloader_seed"]
    df["lr"] = run_sim_args["lr"]
    df["test_batch_size"] = 4096  # hardcoded in load_housing
    df["test_fraction"] = 0.2  # hardcoded in load_housing
    df["epsilon"] = epsilon

    # Save to CSV
    if not debug:
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    plotters.plot_housing_results(
        all_test_losses=all_test_losses,
        num_steps=num_steps,
        details=details,
        experiment_properties=current_experiment_properties,
        debug=debug,
    )


def main():
    parser = argparse.ArgumentParser(description="Run housing simulation experiment.")
    parser.add_argument("--nb_nodes", type=int, default=10, help="Number of nodes")
    parser.add_argument("--graph_name", type=str, default="expander", help="Graph name")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Privacy budget")
    parser.add_argument(
        "--num_repetition",
        type=int,
        default=5,
        help="Number of repetitions (full passes over data)",
    )
    parser.add_argument(
        "--dataloader_seed", type=int, default=421, help="Seed for dataloader"
    )
    parser.add_argument(
        "--use_optimals",
        action="store_true",
        default=False,
        help="Use optimal workload matrices",
    )
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate")
    parser.add_argument(
        "--nb_micro_batches",
        type=int,
        default=16,
        help="Number of micro-batches per node (default: 16)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run in debug mode (no multiprocessing, no CSV save)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        default=False,
        help="Restart all experiments from scratch. Will ignore already-computed csv files, disabling checkpointing.",
    )

    args = parser.parse_args()

    run_experiment(
        use_optimals=args.use_optimals,
        nb_nodes=args.nb_nodes,
        graph_name=args.graph_name,
        num_repetition=args.num_repetition,
        epsilon=args.epsilon,
        dataloader_seed=args.dataloader_seed,
        lr=args.lr,
        nb_micro_batches=args.nb_micro_batches,
        debug=args.debug,
        recompute=args.recompute,
    )


if __name__ == "__main__":
    torch.manual_seed(421)
    np.random.seed(421)
    # random.seed(421)

    main()
