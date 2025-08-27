import numpy as np
import optimal_factorization
import torch
import utils
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


def build_participation_matrix(nb_steps: int, participation_interval: int):
    assert nb_steps % participation_interval == 0
    m = np.zeros(shape=(nb_steps, participation_interval))

    # Single participation
    if participation_interval >= nb_steps:
        print("Found single participation setting")  # TODO: remove debug print
        for i in range(nb_steps):
            m[i, i] = 1
        return m

    # Multiple round participation
    row_indexes = np.array(
        [participation_interval * i for i in range(nb_steps // participation_interval)]
    )
    for i in range(participation_interval):
        m[row_indexes + i, i] = 1
    return m


def compute_sensitivity(
    C: np.ndarray, participation_interval: int, nb_steps: int
) -> float:
    # Check if C is the zero matrix
    if np.allclose(C, 0):
        return np.inf

    X = C.T @ C

    assert (
        nb_steps % participation_interval == 0
    ), f"Participation Interval {participation_interval} does not divide number of steps {nb_steps}"

    participation_mask = utils.get_orthogonal_mask(n=nb_steps, epochs=nb_steps)

    if np.all(X >= 0):
        # Using the trick of Corollary 2.1 (https://proceedings.mlr.press/v202/choquette-choo23a/choquette-choo23a.pdf)
        contrib_matrix = build_participation_matrix(
            nb_steps=nb_steps, participation_interval=participation_interval
        )
        sens = np.sqrt(np.max(np.diag(contrib_matrix.T @ X @ contrib_matrix)))

    elif np.all((1 - participation_mask) * X == 0):
        diag = np.diag(X)
        sensitivities = []
        for i in range(participation_interval):
            idx = np.arange(i, nb_steps, participation_interval)
            sensitivities.append(np.sqrt(np.sum(diag[idx] ** 2)))
        sens = np.max(np.array(sensitivities))

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


def build_local_DL_workload(matrix: np.ndarray, nb_steps: int, initial_power: int = 0):
    """Builds the local version of the DL workload, where each node will have the same local correlation.
    This is the workload that will be optimized in order to obtain an optimal C such that A = B @ pi@ np.kron(In, C)

    Args:
        matrix (np.ndarray): The graph matrix
        nb_steps (int): The number of iterations of DL (and communication rounds)
        initial_power (int, default 0): Initial power of the workload matrix.
            Defines what power of matrix is in the diagonal.
            1 is the matrix itself (optimization workload), 0 is Id (privacy workload).
    """
    nb_nodes = matrix.shape[0]

    dl_workload = build_DL_workload(
        matrix, nb_steps=nb_steps, initial_power=initial_power
    )  # nT * nT
    pi = get_pi(nb_nodes=nb_nodes, nb_iterations=nb_steps)  # nT * Tn
    A = dl_workload @ pi

    gram_workload = np.zeros((nb_steps, nb_steps))
    for i in range(nb_nodes):
        Ai = A[:, nb_steps * i : nb_steps * (i + 1)]
        gram_workload += Ai.T @ Ai

    # TODO: remove this
    is_positive_definite = utils.check_positive_definite(gram_workload)

    if is_positive_definite:
        surrogate_workload = np.linalg.cholesky(gram_workload)
    else:
        print(
            "!" * 50
            + "Warning - DL surrogate workload is not definite positive, trying an experimental surrogate workload that may make optimization fail."
            + "!" * 50
        )
        # Compute square root using - but problem if this workload is not full rank?
        eigvals, eigvecs = np.linalg.eigh(gram_workload)
        # Set negative eigenvalues to zero (for numerical stability)
        eigvals[eigvals < 0] = 0
        surrogate_workload = eigvecs @ np.diag(np.sqrt(eigvals))

    return surrogate_workload


def MF_OPTIMAL_DL(communication_matrix, nb_nodes, nb_steps, nb_epochs):
    surrogate_workload = build_local_DL_workload(
        communication_matrix, nb_steps=nb_steps
    )
    B_optimal, C_optimal = optimal_factorization.get_optimal_factorization(
        surrogate_workload, nb_steps=nb_steps, nb_epochs=nb_epochs
    )
    return B_optimal, C_optimal


def MF_OPTIMAL_local(communication_matrix, nb_nodes, nb_steps, nb_epochs):
    centralized_workload = np.tri(nb_steps)
    B_optimal, C_optimal = optimal_factorization.get_optimal_factorization(
        centralized_workload, nb_steps=nb_steps, nb_epochs=nb_epochs
    )
    return B_optimal, C_optimal


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
