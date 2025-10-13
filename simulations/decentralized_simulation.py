import argparse
import concurrent.futures
import datetime
import functools
import multiprocessing
import os
import time
import warnings

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
from femnist import femnist_model_initializer, load_femnist
from housing import housing_model_initializer, load_housing
from mfdl_optimizer import MFDLSGD, MFDLSGD_Lazy
from opacus import GradSampleModule
from sklearn.metrics import mean_squared_error
from utils import GraphName, get_communication_matrix, get_graph, time_execution
from workloads_generator import compute_sensitivity

# Remove warnings for Housing that should be safe, raised because of Opacus + Housing
warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients.",
)


def get_nb_elements(dataset_name, test_train_fraction):
    match dataset_name:
        case "housing":
            return int(np.ceil(20640 * test_train_fraction))
        case "femnist":
            return 697932
        case _:
            raise ValueError(f"Unknown dataset {dataset_name}")


def get_datasets_and_model_initializer(dataset_name, seed, nb_nodes, nb_big_batches):
    match dataset_name:
        case "housing":
            trainloaders, testloader = load_housing(
                nb_nodes,
                test_fraction=0.2,
                nb_batches=nb_big_batches,
                seed=seed,
            )
            model_initializer = housing_model_initializer
            loss_function = nn.MSELoss()
        case "femnist":
            trainloaders, testloader = load_femnist(
                total_nodes=nb_nodes, nb_batches=nb_big_batches, seed=seed
            )
            model_initializer = femnist_model_initializer
            loss_function = nn.CrossEntropyLoss()
        case _:
            raise ValueError(f"Unexpected dataset name {dataset_name}")
    return trainloaders, testloader, model_initializer, loss_function


def learning_step(
    models: list,
    loss_function,
    nb_micro_batches_per_step: int,
    optimizers: list,
    data_iters: list,
    dataloaders: list,
    device: torch.device = torch.device("cpu"),
):
    nb_resets = 0
    # Local update
    for node, model in enumerate(models):
        # print(f"Node {node}", end="\r")
        optimizer: MFDLSGD = optimizers[node]
        for _ in range(nb_micro_batches_per_step):
            try:
                batch = next(data_iters[node])
            except StopIteration:
                # Reset the current dataloader.
                data_iters[node] = iter(dataloaders[node])
                batch = next(data_iters[node])
                if node == 0:
                    nb_resets += 1
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_microbatch_grad()
            pred = model(x)

            # For regression, ensure pred and y are 1D
            if isinstance(loss_function, nn.MSELoss):
                pred = pred.squeeze()
                y = y.squeeze()
            loss = loss_function(pred, y)
            loss.backward()
            optimizer.microbatch_step()
            del loss, pred, x, y
            torch.cuda.empty_cache()
        optimizer.step()
        optimizer.zero_grad()
    return nb_resets


# @time_execution
@torch.no_grad()
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
    mu: float,
    micro_batches_per_step: int,
    trainloaders: list[data.DataLoader],
    testloader: data.DataLoader,
    model_initializer,
    loss_function,
    test_eval_interval: int,
    lr: float = 1e-1,
    device: torch.device = torch.device("cpu"),
    seed: int = 421,
):
    nb_resets = 0
    num_nodes = graph.number_of_nodes()
    # Ensure all models start from the same initial weights by seeding before each model creation
    models = []
    for _ in range(num_nodes):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = GradSampleModule(model_initializer(seed)).to(device)
        models.append(model)
    participations_intervals = [
        len(trainloader) // micro_batches_per_step for trainloader in trainloaders
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
                noise_seed=seed,
                lr=lr,
                device=device,
                id=i,
                noise_multiplier=1 / mu,
                batch_size=batch_size * micro_batches_per_step,
            )
        )
    data_iters = [iter(trainloaders[i]) for i in range(num_nodes)]

    test_losses: list[list[float]] = [[] for _ in range(num_nodes)]
    train_losses: list[list[float]] = [[] for _ in range(num_nodes)]
    test_accs: list[list[float]] = [[] for _ in range(num_nodes)]

    step_times = []
    for step in range(num_steps):
        # Compute values for print
        if step_times:
            avg_time = sum(step_times) / len(step_times)
            eta = avg_time * (num_steps - (step + 1))
            step_time_str = f"{step_times[-1]:.2f}s"
            avg_time_str = f"{avg_time:.2f}s/step"
            eta_str = str(datetime.timedelta(seconds=int(eta)))
        else:
            step_time_str = "???"
            avg_time_str = "???"
            eta_str = "???"
        print(
            f"Step {step + 1:>{len(str(num_steps))}}/{num_steps} | "
            f"Step time: {step_time_str} | Avg: {avg_time_str} | ETA: {eta_str}"
        )
        start_step_time = time.time()
        nb_resets += learning_step(
            models,
            loss_function=loss_function,
            nb_micro_batches_per_step=micro_batches_per_step,
            optimizers=optimizers,
            data_iters=data_iters,
            dataloaders=trainloaders,
            device=device,
        )
        average_step(graph, models)
        # Time evaluation of train and test losses
        eval_start_time = time.time()
        for node_id, model in enumerate(models):
            # print(f"Evaluating trainset {node_id}")
            loss_list, _ = evaluate_models(
                [model], trainloaders[node_id], device, loss_function
            )
            loss = loss_list[0]
            train_losses[node_id].append(loss)

        # Compute the test loss only every test_eval_interval steps
        if (step + 1) % test_eval_interval == 0 or step == num_steps - 1:
            print(f"Evaluating test set:")
            step_test_losses, step_test_acc = evaluate_models(
                models, testloader, device, loss_function, False, is_testset=True
            )
            eval_end_time = time.time()
            eval_duration = eval_end_time - eval_start_time
            print(
                f"Evaluation took {eval_duration:.2f} seconds. Average test acc: {np.mean(step_test_acc)*100:.2f}%"
            )
        else:
            # Fill with None or previous value if you want to keep shape
            step_test_losses = [np.NaN for _ in range(len(models))]
            step_test_acc = [np.NaN for _ in range(len(models))]

        for node_id, loss in enumerate(step_test_losses):
            test_losses[node_id].append(loss)
        for node_id, acc in enumerate(step_test_acc):
            test_accs[node_id].append(acc)
        end_step_time = time.time()
        step_duration = end_step_time - start_step_time
        step_times.append(step_duration)

    # Losses of shape [nb_steps, nb_nodes]. Time is the 1st axis, nodes the second.
    res_test_losses = np.array(test_losses).T
    res_train_losses = np.array(train_losses).T
    res_test_acc = np.array(test_accs).T
    print(f"Simulation needed {nb_resets} passes through the dataset")
    return models, res_test_losses, res_train_losses, res_test_acc


@torch.no_grad()
def evaluate_models(
    models: list, dataloader, device, loss_function, debug=False, is_testset=False
):
    avg_loss_list = []
    acc_list = []
    is_classification = is_testset and isinstance(loss_function, nn.CrossEntropyLoss)
    for node_id, model in enumerate(models):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        start_time = time.time()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            if isinstance(loss_function, nn.MSELoss):
                output = output.squeeze()
            batch_size = x.size(0)
            loss = loss_function(output, y)
            total_loss += loss.item() * batch_size
            if is_classification:
                pred_labels = output.argmax(dim=1)
                correct_predictions += (pred_labels == y).sum().item()
            total_samples += batch_size
        avg_loss = total_loss / total_samples if total_samples > 0 else float("nan")
        avg_loss_list.append(avg_loss)
        if is_classification:
            accuracy = (
                correct_predictions / total_samples
                if total_samples > 0
                else float("nan")
            )
            acc_list.append(accuracy)
        end_time = time.time()
        if debug:
            print(
                f"Node {node_id} evaluation took {end_time - start_time:.2f} seconds."
            )
        model.train()
    if is_classification:
        return avg_loss_list, acc_list
    else:
        return avg_loss_list, [np.NaN for _ in models]


def run_simulation(
    args,  # Tuple (name, C) for the processpoolexecutor map
    G: nx.Graph,
    num_steps,
    micro_batches_per_step,
    mu: float,
    dataset_name,
    device_name: str,
    nb_micro_batches,
    dataloader_seed,
    lr: float,
    nb_batches,
):
    np.random.seed(dataloader_seed)
    torch.manual_seed(dataloader_seed)
    device = torch.device(device_name)
    n = G.number_of_nodes()
    trainloaders, testloader, model_initializer, loss_function = (
        get_datasets_and_model_initializer(
            dataset_name,
            seed=dataloader_seed,
            nb_nodes=n,
            nb_big_batches=nb_batches,
        )
    )
    name, C = args
    print(f"Running simulation for {name} (PID: {os.getpid()})")
    sens = compute_sensitivity(
        C,
        participation_interval=nb_micro_batches // micro_batches_per_step,
        nb_steps=num_steps,
    )
    print(f"sens²(C_{name}) = {sens**2}")

    start_time = time.time()
    _, test_losses, train_losses, test_accs = run_decentralized_training(
        G,
        num_steps=num_steps,
        C=C,
        mu=mu,
        micro_batches_per_step=micro_batches_per_step,
        trainloaders=trainloaders,
        testloader=testloader,
        model_initializer=model_initializer,
        loss_function=loss_function,
        lr=lr,
        test_eval_interval=nb_batches,  # One eval test/epoch
        device=device,
        seed=dataloader_seed,
    )
    elapsed_time = time.time() - start_time
    print(f"{name}: Training took {elapsed_time:.2f} seconds.")
    return name, test_losses, train_losses, test_accs


def run_experiment(
    debug=True,
    use_optimals=False,
    nb_nodes=10,
    graph_name: GraphName = "expander",
    mu: float = 1.0,
    num_repetition=5,
    dataset_name="housing",
    seed=421,
    lr=1e-1,
    nb_big_batches=16,
    recompute: bool = False,
    pre_cache: bool = False,
):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    # device_name = "cpu"

    device = torch.device(device_name)
    print(f"Device : {device}")

    G = get_graph(graph_name, nb_nodes, seed=seed)
    nb_nodes = G.number_of_nodes()
    communication_matrix = get_communication_matrix(G)

    micro_batches_per_step = 1

    trainloaders, _, _, _ = get_datasets_and_model_initializer(
        dataset_name, seed, nb_nodes, nb_big_batches
    )

    nb_micro_batches_list = [len(loader) for loader in trainloaders]
    del trainloaders  # Memory optimization
    print(nb_micro_batches_list)
    assert np.all(np.array(nb_micro_batches_list) == np.min(nb_micro_batches_list))
    nb_micro_batches_val = np.min(nb_micro_batches_list)

    num_steps = num_repetition * (nb_micro_batches_val // micro_batches_per_step)

    print(f"Number of steps per epoch: {num_steps // num_repetition}")

    # String of experiment details for plots
    current_experiment_properties = (
        f"graph{graph_name}_"
        f"nodes{nb_nodes}_"
        f"mu{mu}_"
        f"reps{num_repetition}_"
        f"nbbatches{nb_big_batches}_"
        f"mbps{micro_batches_per_step}_"
        f"lr{lr}_"
        f"seed{seed}_"
        f"steps{num_steps}_"  # Should be redundant
    )
    details = (
        f"Dataset: {dataset_name} | "
        f"Graph: {graph_name} | "
        f"Nb nodes: {nb_nodes} | "
        f"Micro-batches per step: {micro_batches_per_step} | "
        f"Nb big batches: {nb_big_batches} | "
        f"Num_repetitions: {num_repetition}"
    )
    csv_path = f"results/{dataset_name}/simulation_{current_experiment_properties}.csv"

    # Check if results already exist and skip if not recomputing
    if (not recompute) and os.path.exists(csv_path):
        print(f"Results already exist at {csv_path}. Skipping computation.")
        return

    # Here, we only consider "local" correlations, hence nb_nodes = 1.
    C_NONOISE = np.zeros((num_steps, num_steps))
    C_LDP = workloads_generator.MF_LDP(nb_nodes=1, nb_iterations=num_steps)
    C_ANTIPGD = workloads_generator.MF_ANTIPGD(nb_nodes=1, nb_iterations=num_steps)
    C_SR_LOCAL = workloads_generator.SR_local_factorization(nb_iterations=num_steps)
    C_BSR_LOCAL = workloads_generator.BSR_local_factorization(
        nb_iterations=num_steps, nb_epochs=num_repetition
    )

    all_test_losses = {}
    all_train_losses = {}
    all_test_accs = {}

    configs = [
        ("Unnoised baseline", C_NONOISE),
        ("LDP", C_LDP),
        ("ANTIPGD", C_ANTIPGD),
        # ("BSR_LOCAL", C_SR_LOCAL),
        ("BSR_BANDED_LOCAL", C_BSR_LOCAL),
    ]

    if use_optimals:
        print("Starting computation of C_OPTIMAL_LOCAL...")
        start_time_local = time.time()
        C_OPTIMAL_LOCAL = workloads_generator.MF_OPTIMAL_local(
            communication_matrix=communication_matrix,
            nb_nodes=nb_nodes,
            nb_steps=num_steps,
            nb_epochs=num_repetition,
            caching=True,
            verbose=True,
        )
        elapsed_time_local = time.time() - start_time_local
        print(
            f"Finished computation of C_OPTIMAL_LOCAL in {elapsed_time_local:.2f} seconds."
        )

        print("Starting computation of C_OPTIMAL_MSG...")
        start_time_dl = time.time()
        C_OPTIMAL_DL = workloads_generator.MF_OPTIMAL_DL(
            communication_matrix=communication_matrix,
            nb_nodes=nb_nodes,
            nb_steps=num_steps,
            nb_epochs=num_repetition,
            post_average=False,
            graph_name=graph_name,
            seed=seed,
            caching=True,
            verbose=True,
        )
        elapsed_time_dl = time.time() - start_time_dl
        print(f"Finished computation of C_OPTIMAL_DL in {elapsed_time_dl:.2f} seconds.")

        print("Starting computation of C_OPTIMAL_DL...")
        start_time_dl = time.time()
        C_OPTIMAL_DL_POSTAVERAGE = workloads_generator.MF_OPTIMAL_DL(
            communication_matrix=communication_matrix,
            nb_nodes=nb_nodes,
            nb_steps=num_steps,
            nb_epochs=num_repetition,
            post_average=True,
            graph_name=graph_name,
            seed=seed,
            caching=True,
            verbose=True,
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
            C_OPTIMAL_DL_POSTAVERAGE,
            title="C optimal DL averaged workload",
            details=details,
            save_name_properties=f"DLAVGopti_{current_experiment_properties}",
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
        compute_sensitivity(
            C_OPTIMAL_DL_POSTAVERAGE,
            participation_interval=num_steps // num_repetition,
            nb_steps=num_steps,
        )
        configs.append(("OPTIMAL_DL_MSG", C_OPTIMAL_DL))
        configs.append(("OPTIMAL_DL_POSTAVG", C_OPTIMAL_DL_POSTAVERAGE))
        configs.append(("OPTIMAL_LOCAL", C_OPTIMAL_LOCAL))

    if pre_cache:
        # Don't run simulations, useful for servers where factorization is quick but simulation slow.
        return

    # Use ProcessPoolExecutor for true parallelism with PyTorch
    run_sim_args = {
        "G": G,
        "num_steps": num_steps,
        "micro_batches_per_step": micro_batches_per_step,
        "mu": mu,
        "dataset_name": dataset_name,
        "device_name": device_name,
        "nb_micro_batches": nb_micro_batches_val,
        "dataloader_seed": seed,
        "lr": lr,
        "nb_batches": nb_big_batches,
    }
    run_simulation_partial = functools.partial(run_simulation, **run_sim_args)

    if True:
        results = []
        for config in configs:
            results.append(run_simulation_partial(config))
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(mp_context=context) as executor:
            results = list(executor.map(run_simulation_partial, configs))

    for name, test_losses, train_losses, test_accs in results:
        all_test_losses[name] = test_losses
        all_train_losses[name] = train_losses
        all_test_accs[name] = test_accs

    # Organize results into a DataFrame
    records = []
    for name in all_test_losses.keys():
        test_losses = all_test_losses[name]
        train_losses = all_train_losses[name]
        test_accs = all_test_accs[name]
        for step in range(test_losses.shape[0]):
            for node in range(test_losses.shape[1]):
                records.append(
                    {
                        "method": name,
                        "step": step,
                        "node": node,
                        "test_loss": test_losses[step, node],
                        "train_loss": train_losses[step, node],
                        "test_acc": test_accs[step, node],
                    }
                )
    df = pd.DataFrame(records)
    # Add experiment parameters to the DataFrame for each record
    df["graph_name"] = graph_name
    df["num_passes"] = num_repetition
    df["total_steps"] = num_steps
    df["num_nodes"] = nb_nodes
    df["num_repetitions"] = num_repetition
    df["micro_batches_per_step"] = micro_batches_per_step
    df["nb_micro_batches"] = nb_micro_batches_val
    df["nb_big_batches"] = nb_big_batches
    df["device"] = device_name
    df["dataloader_seed"] = run_sim_args["dataloader_seed"]
    df["lr"] = run_sim_args["lr"]
    df["mu"] = mu
    df["dataset_name"] = dataset_name

    # Save to CSV
    if not debug:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
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
    parser.add_argument("--mu", type=float, default=1.0, help="Privacy budget")
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
        "--nb_batches",
        type=int,
        default=16,
        help="Number of (big) batches per node (default: 16)",
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
    parser.add_argument(
        "--pre_cache",
        action="store_true",
        default=False,
        help="Only compute the necessary matrix factorizations, and make sure they are in the cache for later simulations.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="housing",
        help="Dataset name. Should be `housing` or `femnist`",
    )

    args = parser.parse_args()

    run_experiment(
        use_optimals=args.use_optimals,
        nb_nodes=args.nb_nodes,
        graph_name=args.graph_name,
        num_repetition=args.num_repetition,
        mu=args.mu,
        seed=args.dataloader_seed,
        lr=args.lr,
        nb_big_batches=args.nb_batches,
        debug=args.debug,
        recompute=args.recompute,
        pre_cache=args.pre_cache,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()
