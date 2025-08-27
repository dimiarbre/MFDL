import numpy as np
import torch
from scipy.linalg import toeplitz


def get_pi(nb_nodes, nb_iterations):
    """
    Generates Pi in the paper: converts a node-wise indexing to the general time-wise index.
    Uses advanced indexing for efficiency.
    """
    if nb_nodes == 0 or nb_iterations == 0:
        raise ValueError("0-dimensional permutation is not allowed")
    permutation = np.zeros((nb_nodes * nb_iterations, nb_nodes * nb_iterations))
    for t in range(nb_iterations):
        for i in range(nb_nodes):
            permutation[t * nb_nodes + i, i * nb_iterations + t] = 1
    return permutation


def build_participation_matrix(num_epochs: int, participation_interval: int):
    assert num_epochs % participation_interval == 0
    m = np.zeros(shape=(num_epochs, participation_interval))

    # Single participation
    if participation_interval >= num_epochs:
        print("Found single participation setting")  # TODO: remove debug print
        for i in range(num_epochs):
            m[i, i] = 1
        return m

    # Multiple round participation
    row_indexes = np.array(
        [
            participation_interval * i
            for i in range(num_epochs // participation_interval)
        ]
    )
    for i in range(participation_interval):
        m[row_indexes + i, i] = 1
    return m


def compute_sensitivity(
    C: np.ndarray, participation_interval: int, num_epochs: int
) -> float:
    # Check if C is the zero matrix
    if np.allclose(C, 0):
        return np.inf

    X = C.T @ C

    if np.all(X >= 0):
        contrib_matrix = build_participation_matrix(
            num_epochs=num_epochs, participation_interval=participation_interval
        )
        sens = np.sqrt(np.max(np.diag(contrib_matrix.T @ X @ contrib_matrix)))

    else:
        raise NotImplementedError("Negative matrix factorization")
    return sens


def MF_LDP(nb_nodes, nb_iterations):
    C = np.identity(nb_nodes * nb_iterations)
    return C


def MF_ANTIPGD(nb_nodes, nb_iterations):
    C_local = np.tril(np.ones((nb_iterations, nb_iterations)))
    # Use np.kron to create a block diagonal matrix efficiently
    C_global = np.kron(np.eye(nb_nodes), C_local)
    return C_global


def build_DL_workload(
    matrix: np.ndarray, nb_steps: int, initial_power: int = 0
) -> np.ndarray:
    """Creates the decentralized learning workload from a given matrix.
    Replication keeps spatial structure (e.g. a block in the matrix is a state of the system).

    Args:
        matrix (np.ndarray): the gossip matrix, dimension (n,n)
        nb_steps (int): number of steps to simulate
        initial_power (int, default 0): Initial power of the workload matrix.
            Defines what power of matrix is in the diagonal.
            1 is the matrix itself (optimization workload), 0 is Id (privacy workload).

    Outputs:
        time_matrix (np.ndarray): the stacked gossip matrix through time, dimension (n*nb_steps,n*nb_steps)
    """
    n = matrix.shape[0]
    time_matrix = np.zeros((n * nb_steps, n * nb_steps))
    for i in range(nb_steps):
        time_matrix += np.kron(
            np.eye(nb_steps, nb_steps, -i),
            np.linalg.matrix_power(matrix, i + initial_power),
        )
    return time_matrix


def space_to_time_permutation_matrix(nb_steps: int, nb_nodes: int):
    """
    Generates a permutation matrix that goes from a spatial repartition (n*T) to a temporal repartion (T*n).
    It returns a matrix Pi. If X is a vector composed of nb_nodes blocks of nb_steps values, then Pi @ X is a permutation of this vector composed of nb_steps blocks of nb_nodes values
    """
    pi = np.zeros((nb_nodes * nb_steps, nb_steps * nb_nodes))

    for i in range(nb_nodes):
        for t in range(nb_steps):
            pi[nb_nodes * t + i][nb_steps * i + t] = 1
    return pi


def BSR_local_factorization(nb_iterations):
    """Code from https://github.com/npkalinin/Matrix-Factorization-DP-Training"""

    # Workload without momentum
    workload_tensor = torch.ones(nb_iterations)

    # Square root computation
    y = torch.zeros_like(workload_tensor)
    y[0] = torch.sqrt(workload_tensor[0])
    for k in range(1, len(workload_tensor)):
        y[k] = (workload_tensor[k] - torch.dot(y[1:k], y[1:k].flip(0))) / (2 * y[0])

    #
    C = toeplitz(y)
    return C
