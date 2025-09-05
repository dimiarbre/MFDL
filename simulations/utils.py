import time
import tracemalloc
from typing import Literal

import networkx as nx
import numpy as np


def expander_graph(n, d):
    if d < n:
        G = nx.random_regular_graph(d, n)
    else:
        raise ValueError(
            "Degree d must be less than number of nodes n for a regular graph."
        )
    return G


GraphName = Literal[
    "expander",
    "empty",
    "cycle",
    "complete",
    "erdos",
    "grid",
    "star",
    "florentine",
    "ego",
    "chain",
]


def get_graph(name: GraphName, n: int, seed) -> nx.Graph:
    G: nx.Graph
    assert n >= 0, f"Should not make star graph with {n} nodes"
    match name:
        case "expander":
            if n == 1:
                G = nx.empty_graph(n)
            else:
                d = int(np.ceil(np.log(n)))
                # Ensure d is at least 1 and less than n
                d = max(1, min(d, n - 1))
                # Find the largest d < n that divides n
                divisors = [k for k in range(d, 0, -1) if n % k == 0]
                if divisors:
                    d = divisors[0]
                else:
                    d = 1  # fallback to 1 if no divisor found
                    print(
                        f"Could not find a divisor for {n} nodes, falling back to 1 node."
                    )
                assert d < n, "Degree d must be less than number of nodes n"
                assert n % d == 0, "Degree d must divide n"
                G = expander_graph(n, d)
        case "empty":
            G = nx.empty_graph(n)
        case "cycle":
            G = nx.cycle_graph(n)
        case "complete":
            G = nx.complete_graph(n)
        case "erdos":
            G = nx.erdos_renyi_graph(n, np.log(n) / n, seed=seed)
        case "grid":
            if int(np.sqrt(n)) ** 2 != n:
                raise ValueError(
                    f"Grid graph requires n to be a perfect square, got n={n}"
                )
            side = int(np.sqrt(n))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)
        case "star":
            if n == 1:
                G = nx.empty_graph(n)
            else:
                G = nx.star_graph(n - 1)
        case "florentine":
            G = nx.florentine_families_graph()
            # Convert indexes to int for florentine graph
            # print(f"Number of nodes: {G.number_of_nodes()}")
            G = nx.convert_node_labels_to_integers(G)
            # print(f"Number of nodes: {G.number_of_nodes()}")
        case "ego":
            name_edgelist = "graphs/facebook/" + str(414) + ".edges"
            my_graph = nx.read_edgelist(name_edgelist)
            my_graph = nx.relabel_nodes(my_graph, lambda x: int(x))
            Gcc = sorted(nx.connected_components(my_graph), key=len, reverse=True)
            G = my_graph.subgraph(Gcc[0]).copy()
            G = nx.convert_node_labels_to_integers(G, label_attribute="fb_id")
        case "chain":
            G = nx.path_graph(n)
        case _:
            raise ValueError(f"Invalid graph name {name}")
    G.add_edges_from(
        [(i, i) for i in range(G.number_of_nodes())]
    )  # Always keep this to make a useful graph
    return G


def graph_require_seed(graph_name: GraphName) -> bool:
    if graph_name in [
        "star",
        "grid",
        "complete",
        "cycle",
        "empty",
        "expander",
        "florentine",
        "ego",
        "chain",
    ]:
        return False
    elif graph_name in [
        "erdos",
    ]:
        return True
    else:
        raise NotImplementedError(
            f"Did not implement wether graph {graph_name} requires a seed"
        )


def get_orthogonal_mask(n: int, epochs: int = 1) -> np.ndarray:
    """Computes a mask that imposes orthogonality constraints on the optimization.

    Args:
        n: the size of the mask
        epochs: The number of epochs

    Returns:
        A 0/1 mask
    """
    mask = np.ones((n, n))
    for i in range(n // epochs):
        mask[i :: n // epochs, i :: n // epochs] = np.eye(epochs)
    return mask


def check_positive_definite(matrix: np.ndarray) -> bool:
    eigvals = np.linalg.eigvalsh(matrix)
    if np.any(eigvals < -1e-8):
        return False
    else:
        return True


def get_communication_matrix(G: nx.Graph) -> np.ndarray:
    matrix: np.ndarray = nx.to_numpy_array(G)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    assert not np.isnan(matrix).any(), "Communication matrix contains NaN values"
    return matrix


def profile_memory_usage(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        func_name = func.__name__
        print(f"Profiled function: {func_name}")
        print(
            f"Current memory usage: {current / 10**6:.3f} MB; Peak: {peak / 10**6:.3f} MB"
        )
        print(f"Execution time: {end_time - start_time:.6f} seconds")
        tracemalloc.stop()
        return result

    return wrapper
