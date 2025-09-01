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


def get_graph(name: str, n: int) -> nx.Graph:
    G: nx.Graph
    match name:
        case "expander":
            if n == 1:
                G = nx.empty_graph(n)
            else:
                d = int(np.ceil(np.log(n)))
                # Ensure d divides n and d < n
                if d >= n:
                    d = n - 1
                if n % d != 0:
                    # Find the largest d < n that divides n
                    divisors = [k for k in range(d, 0, -1) if n % k == 0]
                    if not divisors:
                        raise ValueError(
                            f"Cannot find a suitable degree for expander graph with n={n}"
                        )
                    d = divisors[0]
                assert d < n, "Degree d must be less than number of nodes n"
                assert n % d == 0, "Degree d must divide n"
                G = expander_graph(n, d)
        case "empty":
            G = nx.empty_graph(n)
        case "cycle":
            G = nx.cycle_graph(n)
        case "complete":
            G = nx.complete_graph(n)
        case _:
            raise ValueError(f"Invalid graph name {name}")
    G.add_edges_from(
        [(i, i) for i in range(G.number_of_nodes())]
    )  # Always keep this to make a useful graph
    return G


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
