import os

import numpy as np
import optimal_factorization
import torch
import utils
from scipy.linalg import toeplitz
from utils import graph_require_seed, profile_memory_usage


def get_pi(nb_nodes, nb_iterations):
    """
    Generates Pi in the paper: converts a node-wise indexing to the general time-wise index.
    """
    if nb_nodes == 0 or nb_iterations == 0:
        raise ValueError("0-dimensional permutation is not allowed")
    permutation = np.zeros((nb_nodes * nb_iterations, nb_nodes * nb_iterations))
    for t in range(nb_iterations):
        for i in range(nb_nodes):
            permutation[t * nb_nodes + i, i * nb_iterations + t] = 1

    # Check if permutation is a valid permutation matrix
    if not (
        np.all(permutation.sum(axis=0) == 1)
        and np.all(permutation.sum(axis=1) == 1)
        and np.all((permutation == 0) | (permutation == 1))
    ):
        raise ValueError("Generated matrix is not a valid permutation matrix.")

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
    """Computes sens(C), assuming cyclic participation

    Args:
        C (np.ndarray): Encoder matrix
        participation_interval (int): Participation interval (number of batches in an epoch)
        nb_steps (int): Total number of steps, should be participation_interval * num_repetitions

    Returns:
        float: The numeric value of the sensitivity.
    """
    # Check if C is the zero matrix
    if np.allclose(C, 0):
        return np.inf

    X = C.T @ C

    assert (
        nb_steps % participation_interval == 0
    ), f"Participation Interval {participation_interval} does not divide number of steps {nb_steps}"

    nb_epochs = nb_steps // participation_interval
    participation_mask = utils.get_orthogonal_mask(n=nb_steps, epochs=nb_epochs)
    sensitivities = []
    for i in range(participation_interval):
        idx = np.arange(i, nb_steps, participation_interval)
        sensitivities.append(
            np.sqrt(np.sum(np.abs(X[np.ix_(idx, idx)])))
        )  # Upper bound on the sensitivity. Should be tight if the matrix X is positive in all elements
    sens = np.max(np.array(sensitivities))
    return sens
    if np.allclose((1 - participation_mask) * X, 0, atol=1e-10):
        sensitivities = []
        for i in range(participation_interval):
            idx = np.arange(i, nb_steps, participation_interval)
            sensitivities.append(np.sqrt(np.sum(np.abs(X[np.ix_(idx, idx)]))))
        sens = np.max(np.array(sensitivities))
    elif np.all(X >= 0):
        print("Code should never reach here, or one matrix was weird")
        # Using the trick of Corollary 2.1 (https://proceedings.mlr.press/v202/choquette-choo23a/choquette-choo23a.pdf)
        contrib_matrix = build_participation_matrix(
            nb_steps=nb_steps, participation_interval=participation_interval
        )
        sens = np.sqrt(np.max(np.diag(contrib_matrix.T @ X @ contrib_matrix)))

    else:
        raise NotImplementedError("Negative matrix factorization")
    return sens


# TODO: Removed this unused function?
def compute_surrogate_loss(workload, C_inv):
    X = C_inv.T @ C_inv
    return np.trace(X @ workload)


def MF_LDP(nb_nodes, nb_iterations):
    C = np.identity(nb_nodes * nb_iterations)
    return C


def MF_ANTIPGD(nb_nodes, nb_iterations):
    C_local = np.tril(np.ones((nb_iterations, nb_iterations)))
    # Use np.kron to create a block diagonal matrix efficiently
    C_global = np.kron(np.eye(nb_nodes), C_local)
    return C_global


# @profile_memory_usage
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


# @profile_memory_usage
def build_local_DL_gram_workload(
    matrix: np.ndarray, nb_steps: int, initial_power: int = 0
):
    """Builds the local version of the DL Gram workload, where each node will have the same local correlation. Will be used to compute ||A @ np.kron(In, np.pinv(C))||.
    Under this form of correlation, this returns a (smaller) Gram matrix G such that
    ||A @ np.kron(In, np.pinv(C))|| = np.trace(C.T @ G @ C)

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

    return gram_workload


def build_local_DL_workload(
    matrix: np.ndarray,
    nb_steps: int,
    initial_power: int = 0,
    graph_name=None,
    seed=None,
    caching=True,
    verbose=False,
):
    """Builds the local version of the DL workload, where each node will have the same local correlation.
    This is the workload that will be optimized in order to obtain an optimal C such that A = B @ pi@ np.kron(In, C)

    Args:
        matrix (np.ndarray): The graph matrix
        nb_steps (int): The number of iterations of DL (and communication rounds)
        initial_power (int, default 0): Initial power of the workload matrix.
            Defines what power of matrix is in the diagonal.
            1 is the matrix itself (optimization workload), 0 is Id (privacy workload).
        graph_name (optional str): only used for caching, name of the graph
        seed (optional int): seed to generate the graph, only used for caching.
    """
    nb_nodes = len(matrix)
    cache_dir = f"cache/workloads/surrogate_workload/{graph_name}/nodes{nb_nodes}/steps{nb_steps}/"
    cache_filename = f"local_DL_diagonalpower{initial_power}"

    if graph_name is None:
        caching = False
    elif graph_require_seed(graph_name):
        if seed is None:
            caching = False
        else:
            cache_filename += f"_seed{seed}"

    if caching:
        cache_result = get_from_cache(
            cache_dir=cache_dir, filename=cache_filename, verbose=verbose
        )
        if cache_result is not None:
            return cache_result

    gram_workload = build_local_DL_gram_workload(
        matrix=matrix, nb_steps=nb_steps, initial_power=initial_power
    )

    gram_workload_permuted = optimal_factorization._permute_lower_triangle(
        gram_workload
    )

    is_positive_definite = utils.check_positive_definite(gram_workload_permuted)

    if is_positive_definite:
        surrogate_workload = np.linalg.cholesky(gram_workload_permuted)
    else:
        raise NotImplementedError("Non-positive definite Gram workload")
    if caching:
        save_to_cache(
            cache_dir=cache_dir,
            filename=cache_filename,
            matrix=surrogate_workload,
            verbose=verbose,
        )
    return surrogate_workload


def get_from_cache(cache_dir, filename, verbose=False):
    cache_path = os.path.join(cache_dir, filename + ".npy")
    # Try to load from cache
    if os.path.exists(cache_path):
        if verbose:
            print(f"Loading from cache {cache_path}")
        with open(cache_path, "rb") as f:
            C_optimal = np.load(f)
        return C_optimal
    return None


def save_to_cache(cache_dir, filename, matrix: np.ndarray, verbose=False):
    cache_path = os.path.join(cache_dir, filename + ".npy")
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        np.save(f, matrix)
    if verbose:
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"Saved {size_mb:.2f} MB to cache {cache_path} ")


def MF_OPTIMAL_DL(
    communication_matrix,
    nb_nodes,
    nb_steps,
    nb_epochs,
    post_average=False,
    graph_name=None,
    seed=None,
    caching=True,
    verbose=False,
) -> np.ndarray:
    """
    Lazy loader for optimal DL factorization. Caches results to disk for repeated calls.

    Args:
        communication_matrix: np.ndarray, the communication matrix.
        nb_nodes: int, number of nodes.
        nb_steps: int, number of steps.
        nb_epochs: int, number of epochs.
        post_average: bool, whether to use post-averaging.
        graph_name: str, unique name for the graph (required for caching).
        seed: int, optional seed for uniqueness.
        caching: bool, optional, default to True: wether to try and save intermediate workloads to cache.

    Returns:
        C_optimal
    """
    cache_dir = f"cache/correlations_matrixes/{graph_name}/nodes{nb_nodes}/steps{nb_steps}/epochs{nb_epochs}/"
    cache_filename = f"OptimalDL_{'PostAVG' if post_average else "Msg"}"

    if graph_name is None:
        caching = False
    elif graph_require_seed(graph_name):
        if seed is None:
            caching = False
        else:
            cache_filename += f"_seed{seed}"

    if caching:
        cache_result = get_from_cache(
            cache_dir=cache_dir, filename=cache_filename, verbose=verbose
        )
        if cache_result is not None:
            return cache_result

    surrogate_workload = build_local_DL_workload(
        communication_matrix,
        nb_steps=nb_steps,
        initial_power=int(post_average),
        graph_name=graph_name,
        seed=seed,
        verbose=verbose,
    )
    C_optimal = optimal_factorization.get_optimal_factorization(
        surrogate_workload, nb_steps=nb_steps, nb_epochs=nb_epochs, verbose=verbose
    )

    # Save to cache
    if caching:
        save_to_cache(
            cache_dir=cache_dir,
            filename=cache_filename,
            matrix=C_optimal,
            verbose=verbose,
        )

    return C_optimal


def MF_OPTIMAL_local(
    communication_matrix,
    nb_nodes,
    nb_steps,
    nb_epochs,
    caching=True,
    verbose=False,
):
    """
    Computes and caches the optimal local factorization.

    Args:
        communication_matrix: np.ndarray, the communication matrix (unused here).
        nb_nodes: int, number of nodes.
        nb_steps: int, number of steps.
        nb_epochs: int, number of epochs.
        caching: bool, optional, default to True.

    Returns:
        C_optimal: np.ndarray, the optimal factorization matrix.
    """
    cache_dir = f"cache/centralized/steps{nb_steps}/epochs{nb_epochs}"
    cache_filename = f"optimal_local"

    if caching:
        cache_result = get_from_cache(
            cache_dir=cache_dir, filename=cache_filename, verbose=verbose
        )
        if cache_result is not None:
            return cache_result

    centralized_workload = np.tri(nb_steps)
    C_optimal = optimal_factorization.get_optimal_factorization(
        centralized_workload, nb_steps=nb_steps, nb_epochs=nb_epochs, verbose=verbose
    )

    if caching:
        save_to_cache(
            cache_dir=cache_dir,
            filename=cache_filename,
            matrix=C_optimal,
            verbose=verbose,
        )

    return C_optimal


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
