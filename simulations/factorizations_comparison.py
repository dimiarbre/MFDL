import argparse
import os
import time
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotters
import scienceplots
import utils
import workloads_generator
from utils import GraphName

FactorizationName = Literal[
    "LDP",
    "ANTIPGD",
    "BSR_LOCAL",
    "OPTIMAL_LOCAL",
    "OPTIMAL_DL_MSG",
    "OPTIMAL_DL_POSTAVG",
]

POSSIBLE_FACTORIZATION: list[FactorizationName] = [
    "LDP",
    "ANTIPGD",
    "BSR_LOCAL",
    "OPTIMAL_LOCAL",
    "OPTIMAL_DL_MSG",
    "OPTIMAL_DL_POSTAVG",
]


def workload_wrapper(
    workload_name: FactorizationName,
    communication_matrix,
    nb_nodes,
    nb_iterations,
    nb_epochs,
    graph_name=None,
    seed=None,
    verbose=False,
):
    match workload_name:
        case "LDP":
            # Forcing 1 node for local matrix
            return workloads_generator.MF_LDP(nb_nodes=1, nb_iterations=nb_iterations)
        case "ANTIPGD":
            # Forcing 1 node for local matrix
            return workloads_generator.MF_ANTIPGD(
                nb_nodes=1, nb_iterations=nb_iterations
            )
        case "BSR_LOCAL":
            return workloads_generator.BSR_local_factorization(
                nb_iterations=nb_iterations
            )
        case "OPTIMAL_LOCAL":
            return workloads_generator.MF_OPTIMAL_local(
                communication_matrix=communication_matrix,  # Unused argument
                nb_nodes=1,  # Unused argument
                nb_steps=nb_iterations,
                nb_epochs=nb_epochs,
            )
        case "OPTIMAL_DL_MSG":
            return workloads_generator.MF_OPTIMAL_DL(
                communication_matrix=communication_matrix,
                nb_nodes=nb_nodes,
                nb_steps=nb_iterations,
                nb_epochs=nb_epochs,
                post_average=False,
                graph_name=graph_name,
                seed=seed,
                verbose=verbose,
            )
        case "OPTIMAL_DL_POSTAVG":
            return workloads_generator.MF_OPTIMAL_DL(
                communication_matrix=communication_matrix,
                nb_nodes=nb_nodes,
                nb_steps=nb_iterations,
                nb_epochs=nb_epochs,
                post_average=True,
                graph_name=graph_name,
                seed=seed,
                verbose=verbose,
            )
        case _:
            raise NotImplementedError(
                f"Did not find a workload strategy {workload_name}"
            )


def evaluate_loss(gram_workload, C, participation_interval, nb_iterations):
    sens2 = (
        workloads_generator.compute_sensitivity(
            C=C, participation_interval=participation_interval, nb_steps=nb_iterations
        )
        ** 2
    )
    C_inv = np.linalg.pinv(C)
    norm2 = np.trace(C_inv.T @ gram_workload @ C_inv)

    return norm2 * sens2


def run_experiment(
    graph_name: GraphName,
    nb_nodes: int,
    nb_repetition: int,
    participation_interval: int,
    seed: int = 421,
    verbose=False,
):
    """Run experiments for LOCAL node correlation: A = B @ np.kron(In, C), for multiple types of C.

    Args:
        graph_name (str): Type of graph, see utils.get_graph
        nb_nodes (int): Number of nodes in the graph
        nb_repetition (int): Number of passes through the dataset
        participation_interval (int): Interval between to participations (number of batches)
        seed (int): Seed for randomness
        verbose (bool, default False): Print a lot if True.
    """
    np.random.seed(seed)

    # Setup
    G = utils.get_graph(graph_name, nb_nodes, seed=seed)
    assert (
        nb_nodes == G.number_of_nodes()
    ), f"Incorrect number of nodes: expected {nb_nodes}, but a graph of {nb_nodes} was returned"
    communication_matrix = utils.get_communication_matrix(G)

    nb_steps = nb_repetition * participation_interval

    # Build workloads
    if verbose:
        print("Building DL workload")
    gram_message_workload = workloads_generator.build_local_DL_gram_workload(
        matrix=communication_matrix,
        nb_steps=nb_steps,
        initial_power=0,
        verbose=verbose,
        graph_name=graph_name,
        seed=seed,
    )
    if verbose:
        print("Built DL workload")
        print("Building DL optimization workload")
    gram_optimization_workload = workloads_generator.build_local_DL_gram_workload(
        matrix=communication_matrix,
        nb_steps=nb_steps,
        initial_power=1,
        verbose=verbose,
        graph_name=graph_name,
        seed=seed,
    )
    if verbose:
        print("Built DL optimization workload")

    df = pd.DataFrame({})

    for factorization_name in POSSIBLE_FACTORIZATION:
        # TODO: Optimize this, and pass the gram matrix directly to save computation (some factorizations will rebuilt those)
        if verbose:
            print(f"Computing {factorization_name}")
        start_time = time.time()
        C = workload_wrapper(
            factorization_name,
            communication_matrix=communication_matrix,
            nb_nodes=nb_nodes,
            nb_iterations=nb_steps,
            nb_epochs=nb_repetition,
            graph_name=graph_name,
            seed=seed,
            verbose=verbose,
        )
        elapsed_time = time.time() - start_time
        # plotters.plot_factorization(np.linalg.pinv(C), factorization_name + "+ $C^+$")
        sens = workloads_generator.compute_sensitivity(
            C=C, participation_interval=participation_interval, nb_steps=nb_steps
        )
        loss_message = evaluate_loss(
            gram_message_workload,
            C,
            participation_interval=participation_interval,
            nb_iterations=nb_steps,
        )
        loss_optimization = evaluate_loss(
            gram_optimization_workload,
            C,
            participation_interval=participation_interval,
            nb_iterations=nb_steps,
        )
        current_line = {
            "factorization_name": factorization_name,
            "loss_optimization": loss_optimization,
            "loss_message": loss_message,
            "nb_nodes": nb_nodes,
            "graph_name": graph_name,
            "nb_repetition": nb_repetition,
            "participation_interval": participation_interval,
            "seed": seed,
            "factorization_time": elapsed_time,
            "sensitivity": sens,
        }
        df = pd.concat([df, pd.DataFrame([current_line])], ignore_index=True)

    # plt.show()
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run factorization comparison experiments."
    )
    parser.add_argument(
        "--graph_name",
        type=str,
        default="empty",
        help="Type of graph, see utils.get_graph for valid names",
    )
    parser.add_argument(
        "--nb_nodes", type=int, default=10, help="Number of nodes in the graph"
    )
    parser.add_argument(
        "--nb_repetition",
        type=int,
        default=4,
        help="Number of passes through the dataset",
    )
    parser.add_argument(
        "--participation_interval",
        type=int,
        default=16,
        help="Interval between participations (number of batches)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose level",
    )
    parser.add_argument("--seed", type=int, default=421, help="Seed for randomness")

    args = parser.parse_args()

    run_experiment(
        graph_name=args.graph_name,
        nb_nodes=args.nb_nodes,
        nb_repetition=args.nb_repetition,
        participation_interval=args.participation_interval,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    plt.style.use("science")
    main()
