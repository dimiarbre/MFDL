import time
import tracemalloc
import warnings
from typing import Literal

import networkx as nx
import numpy as np
from fedivertex import GraphLoader

warnings.filterwarnings(
    "ignore",
    message="WARNING:absl:WARNING: The JSON-LD `@context` is not standard. Refer to the official @context (e.g., from the example datasets in https://github.com/mlcommons/croissant/tree/main/datasets/1.0). The different keys are: {'samplingRate', 'examples', 'rai'}",
)

# Dictionary to rename methods for display
METHOD_DISPLAY_NAMES = {
    "Unnoised baseline": "Unnoised baseline",
    "LDP": "DP-D-SGD",
    "ANTIPGD": "ANTIPGD",
    "BSR_LOCAL": "SR (Nikita \\& Lampert, 2024)",  # Banded SR, is never used, and thus named wrongly.
    "BSR_BANDED_LOCAL": "BSR (Nikita \\& Lampert, 2024)",  # Banded SR, is never used, and thus named wrongly.
    "OPTIMAL_LOCAL": "D-MF",
    "OPTIMAL_DL_MSG": "Optimal (Message Loss)",
    "OPTIMAL_DL_POSTAVG": "MAFALDA-SGD (Ours)",
    "OPTIMAL_DL_LOCALCOR": "MAFALDA-L-SGD (Ours)",
}

# Dictionary to assign colors to each method
METHOD_COLORS = {
    "Unnoised baseline": "#1f77b4",
    "DP-D-SGD": "#9467bd",
    "ANTIPGD": "#ff7f0e",
    "SR (Nikita \\& Lampert, 2024)": "#e377c2",
    "BSR (Nikita \\& Lampert, 2024)": "#8c564b",
    "D-MF": "#2ca02c",
    "Optimal (Message Loss)": "#17becf",
    "MAFALDA-SGD (Ours)": "#d62728",
}

METHOD_MARKERS = {
    "Unnoised baseline": "o",
    "DP-D-SGD": "s",
    "ANTIPGD": "*",
    "SR (Nikita \\& Lampert, 2024)": "^",
    "BSR (Nikita \\& Lampert, 2024)": "H",
    "D-MF": "<",
    "Optimal (Message Loss)": "X",
    "MAFALDA-SGD (Ours)": ">",
}
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
    "hypercube",
    "peertube",
    "peertube (connex component)",
    "regular",
]

GRAPH_RENAME = {
    "expander": "Expander",
    "erdos": "Erdős-Rényi",
    "peertube (connex component)": "Peertube",
    "peertube": "Peertube (full)",
    "ego": "Facebook Ego",
    "florentine": "Florentine",
}


def expander_graph(n, d, seed):
    if d < n:
        G = nx.random_regular_graph(d, n, seed=seed)
    else:
        raise ValueError(
            "Degree d must be less than number of nodes n for a regular graph."
        )
    return G


def get_erdos_renyi_graph(n, p, seed, total_nb_tries=1000):
    # Be careful to not use the cache if you use this function directly /change the total_nb_tries, as this is not accounted for in the caching scheme for erdos graphs. Consider using get_graph instead.
    connex = False
    nb_tries = 0
    G = nx.empty_graph(n)
    while not connex and nb_tries < total_nb_tries:
        G = nx.erdos_renyi_graph(
            n, p, seed=total_nb_tries * seed + nb_tries
        )  # Have to expand so that seeds don't colide.
        connex = nx.is_connected(G)
        nb_tries += 1
    if nb_tries >= total_nb_tries:
        raise ValueError("Required too long to generate a fully-connected Erdos graph")
    return G


def get_graph(name: GraphName, n: int, seed) -> nx.Graph:
    G: nx.Graph
    assert n >= 0, f"Should not make graph with {n} nodes"
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
                G = expander_graph(n, d, seed)
        case "empty":
            G = nx.empty_graph(n)
        case "cycle":
            G = nx.cycle_graph(n)
        case "complete":
            G = nx.complete_graph(n)
        case "erdos":
            G = get_erdos_renyi_graph(n, np.log(n) / n, seed)
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
        case "florentine":  # 15 nodes
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
        case "peertube":
            loader = GraphLoader()
            G = loader.get_graph(
                software="peertube",
                graph_type="follow",
                index=1,
                disable_tqdm=True,
                only_largest_component=False,
            )
            G = nx.Graph(G)  # Convert to undirected graph
        case "peertube (connex component)":
            loader = GraphLoader()
            G = loader.get_graph(
                software="peertube",
                graph_type="follow",
                index=1,
                disable_tqdm=True,
                only_largest_component=True,
            )
            G = nx.Graph(G)  # Convert to undirected graph
        case "chain":
            G = nx.path_graph(n)
        case "regular":
            # 5 regular graph
            # TODO: Allow user to chose the degree
            G = nx.random_regular_graph(5, n, seed)
        case "hypercube":
            if n <= 0:
                raise ValueError(f"Hypercube graph requires n to be >= 1, got n={n}")
            # use log2 to check that n is a power of two
            log2_n = np.log2(n)
            if abs(round(log2_n) - log2_n) > 1e-12:
                raise ValueError(
                    f"Hypercube graph requires n to be a power of two, got n={n}"
                )
            # compute dimension d such that n == 2**d
            d = int(round(log2_n))
            G = nx.hypercube_graph(d)
        case _:
            raise ValueError(f"Invalid graph name {name}")
    G = nx.convert_node_labels_to_integers(G)
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
        "florentine",
        "ego",
        "chain",
        "peertube",
        "peertube (connex component)",
        "hypercube",
    ]:
        return False
    elif graph_name in ["erdos", "expander", "regular"]:
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


def get_communication_matrix(
    G: nx.Graph, metropolis_hasting: bool = False
) -> np.ndarray:
    matrix: np.ndarray = nx.to_numpy_array(G)
    if metropolis_hasting:
        # Set weights according to Metropolis-Hasting
        n = G.number_of_nodes()
        degrees = matrix.sum(axis=1)
        mh_matrix = np.zeros_like(matrix)
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i, j] > 0:
                    mh_matrix[i, j] = 1 / max(degrees[i], degrees[j])
            mh_matrix[i, i] = 1 - mh_matrix[i].sum()
        matrix = mh_matrix
    else:
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


def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        func_name = func.__name__
        print(f"{func_name}: Execution time: {end_time - start_time:.6f} seconds")
        return result

    return wrapper


def main():
    graph_name: GraphName = "peertube (connex component)"
    G = get_graph(graph_name, 0, 0)
    print(G.number_of_nodes())
    max_degree = max(dict(G.degree()).values())
    print(f"Highest node degree: {max_degree}")
    matrix = get_communication_matrix(G)
    most_connected_node = max(G.degree, key=lambda x: x[1])[0]
    print(f"Most connected node: {most_connected_node}")


if __name__ == "__main__":
    # Only used for debugging purposes, trying out graph loading and stuff.
    main()
