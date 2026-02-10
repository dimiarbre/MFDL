import os
from typing import Optional
from warnings import warn

import numpy as np
import torch
from scipy.linalg import toeplitz

from . import optimal_factorization, utils
from .utils import graph_require_seed, profile_memory_usage


def get_commutation_matrix(nb_nodes, nb_iterations):
    """
    Generates a coomutation matrix S_{nb_nodes, nb_iterations} that goes from a spatial repartition (n*T) to a temporal repartion (T*n).
    It returns a matrix S. If X is a vector composed of nb_nodes blocks of nb_steps values, then S @ X is a permutation of this vector composed of nb_steps blocks of nb_nodes values
    """
    warn(
        "get_commutation_matrix is inefficient and should be replaced by get_commutation_reindexing, which gives the same result with a simple reindexing. See tests/test_workload_generator.py and the documentation of get_commutation_reindexing for examples of how to use the new function.",
        category=DeprecationWarning,
    )
    # TODO: This needs to be redone for optimization purposes. As it stands, this instantiates a big matrix, and we run W @ pi down the line. Since it is a permutation, we could just have a reindexing instead. For instance, with 3 nodes and 3 iterations, we would have something like  pi = [0,3,6,1,4,7,2,5,8], and just compute W[pi]. This would be much more efficient in memory.
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


def get_commutation_reindexing(nb_nodes, nb_iterations):
    """
    Returns a reindexing array for the permutation from spatial (n*T) to temporal (T*n) repartition.
    For example, with 3 nodes and 3 iterations, returns [0, 3, 6, 1, 4, 7, 2, 5, 8].
    Usage:
    pi_old = get_commutation_matrix(nb_nodes, nb_iterations)
    pi, pi_inv = get_commutation_reindexing(nb_nodes, nb_iterations)
    assert np.allclose(W[pi], pi_old @ W) # For all matrixes W.

    Or, if the computation was W @ pi_old, consider instead W[:, pi_inv].
    """
    if nb_nodes == 0 or nb_iterations == 0:
        raise ValueError("0-dimensional permutation is not allowed")
    pi = []
    for t in range(nb_iterations):
        for i in range(nb_nodes):
            pi.append(i * nb_iterations + t)
    pi_np = np.array(pi)
    pi_inv = np.argsort(pi)
    return pi_np, pi_inv


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


def compute_cyclic_repetitions_1node(
    X: np.ndarray, participation_interval: int, nb_steps: int, nb_nodes: int, node: int
) -> float:
    """Compute the sensitivity of a generic workload for a given node.
    This is useful when considering PNDP: if we want to consider epsilon_{a->b}, then we want to explore sensitivity restrained to node a (as it is the only node that can change)

    Args:
        X (np.ndarray): Gram workload for sensitivity (C.T @ B.pinv @ B @ C)
        participation_interval (int): Interval between two participations
        nb_steps (int): Total number of steps in the process
        nb_nodes (int): Number of nodes in the system
        node (int): Node id with regards to whom we compute sensitivity. This is the "victim" node whose dataset might change

    Returns:
        float: Sensitivity for node 'node'
    """
    sensitivities = []
    for t in range(participation_interval):
        # Implement cyclic participation for a given node
        # This corresponds to a participation every participation_interval * nb_nodes in G
        idx = np.arange(
            t * nb_nodes + node,
            nb_steps * nb_nodes,
            participation_interval * nb_nodes,
        )
        sensitivities.append(
            np.sqrt(np.sum(np.abs(X[np.ix_(idx, idx)])))
        )  # Upper bound on the sensitivity. Should be tight if the matrix X is positive in all elements
    sens = np.max(np.array(sensitivities))
    return sens


def compute_cyclic_repetitions(
    X: np.ndarray,
    participation_interval: int,
    nb_steps: int,
    nb_nodes: int,
) -> float:
    """Compute "max(sum(|X_{s,t}|))" under the cyclic participation schema.
    X will mostly be given by other functions, as it allows for optimization in some cases.

    Args:
        X (np.ndarray): Gram workload
        participation_interval (int): Interval of the cycle for a local node
        nb_steps (int): number of total steps of the system
        nb_nodes (int): _description_

    Returns:
        float: The sensitivity of the overall workload, for ALL nodes
    """
    sensitivities = []
    for node in range(nb_nodes):
        sensitivities.append(
            compute_cyclic_repetitions_1node(
                X,
                participation_interval=participation_interval,
                nb_steps=nb_steps,
                nb_nodes=nb_nodes,
                node=node,
            )
        )
    sens = np.max(np.array(sensitivities))
    return sens


def compute_sensitivity(
    C: np.ndarray, participation_interval: int, nb_steps: int
) -> float:
    """Computes sens(C) for a LOCAL encoder matrix C, assuming cyclic participation and square (and invertible) decoder matrix B.

    Args:
        C (np.ndarray): Encoder matrix
        participation_interval (int): Participation interval (number of batches in an epoch)
        nb_steps (int): Total number of steps, should be participation_interval * num_repetitions

    Returns:
        float: The numeric value of the sensitivity.
    """
    assert (
        C.shape[0] == C.shape[1]
    ), f"compute_sensitivity should only consider squared factorization, got shape {C.shape}. For more general cases, use compute_sensitivity_rectangularworkload"
    assert nb_steps == len(
        C
    ), f"C should be of size nb_steps, but got {len(C)} instead of {nb_steps}"
    # TODO: remove the unnecessary argument, and recompute nb_steps manually from this

    # Check if C is the zero matrix
    if np.allclose(C, 0):
        return np.inf

    X = C.T @ C

    assert (
        nb_steps % participation_interval == 0
    ), f"Participation Interval {participation_interval} does not divide number of steps {nb_steps}"

    sens = compute_cyclic_repetitions(X, participation_interval, nb_steps, nb_nodes=1)
    return sens


def compute_sensitivity_rectangularworkload(
    B: np.ndarray,
    C: np.ndarray,
    participation_interval: int,
    nb_steps: int,
) -> float:
    """Computes sens( G -> QCG) for a GLOBAL encoder matrix C, assuming cyclic participation (and same cycle for all nodes).
    This is the formulat max(sum_{s,t}(|C.T @ B^+ @ B @ C|_{s,t})).
    You should prefer compute_sensitivity when considering a square and local workload, as this approach is much less efficient.

    Args:
        B (np.ndarray): Decoder matrix
        C (np.ndarray): Encoder matrix
        participation_interval (int): Participation interval (number of batches in an epoch)
        nb_steps (int): Total number of steps, should be participation_interval * num_repetitions

    Returns:
        float: The numeric value of the sensitivity.
    """
    assert (
        C.shape[1] % nb_steps == 0
    ), "Dimension is not divided by the number of steps."
    nb_nodes = C.shape[1] // nb_steps
    assert (
        nb_steps % participation_interval == 0
    ), f"Participation Interval {participation_interval} does not divide number of steps {nb_steps}"

    # WARNING: this is a hefty computation that will lead to errors.
    X = C.T @ np.linalg.pinv(B) @ B @ C

    return compute_cyclic_repetitions(
        X,
        participation_interval=participation_interval,
        nb_steps=nb_steps,
        nb_nodes=nb_nodes,
    )


def MF_LDP(nb_nodes, nb_iterations):
    C = np.identity(nb_nodes * nb_iterations)
    return C


def MF_ANTIPGD(nb_nodes, nb_iterations):
    C_local = np.tril(np.ones((nb_iterations, nb_iterations)))
    # Use np.kron to create a block diagonal matrix efficiently
    C_global = np.kron(np.eye(nb_nodes), C_local)
    return C_global


# @profile_memory_usage
def build_DL_workload_old(
    matrix: np.ndarray, nb_steps: int, initial_power: int = 0, verbose: bool = False
) -> np.ndarray:
    """Creates the decentralized learning workload from a given matrix.
    Replication keeps spatial structure (e.g. a block in the matrix is a state of the system).
    NB: Old function - should not use this one.

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
        if verbose:
            print(f"Building diagonal {i:0{len(str(nb_steps))}d}/{nb_steps}", end="\r")
        time_matrix += np.kron(
            np.eye(nb_steps, nb_steps, -i),
            np.linalg.matrix_power(matrix, i + initial_power),
        )
    return time_matrix


def build_DL_workload(
    matrix: np.ndarray, nb_steps: int, initial_power: int = 0, verbose: bool = False
) -> np.ndarray:
    """
    Creates the decentralized learning workload from a given matrix.
    The block matrix has W^{initial_power} on the main diagonal,
    W^{initial_power+1} on the first lower diagonal, etc.

    Args:
        matrix (np.ndarray): the gossip matrix, dimension (n,n)
        nb_steps (int): number of steps to simulate
        initial_power (int, default 0): Initial power of the workload matrix.

    Returns:
        time_matrix (np.ndarray): the stacked gossip matrix through time, dimension (n*nb_steps,n*nb_steps)
    """
    n = matrix.shape[0]
    time_matrix = np.zeros((n * nb_steps, n * nb_steps))
    # Precompute powers for efficiency

    for diag in range(nb_steps):
        power = np.linalg.matrix_power(matrix, diag + initial_power)
        for block in range(nb_steps - diag):
            row_start = (block + diag) * n
            row_end = row_start + n
            col_start = block * n
            col_end = col_start + n
            time_matrix[row_start:row_end, col_start:col_end] = power
        if verbose:
            print(f"Filled diagonal {diag+1}/{nb_steps}", end="\r")
    return time_matrix


# @profile_memory_usage
def build_local_DL_gram_workload(
    matrix: np.ndarray,
    nb_steps: int,
    initial_power: int = 0,
    graph_name=None,
    seed=None,
    caching=True,
    verbose=False,
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
    cache_dir = f"cache/workloads/surrogate_workload/{graph_name}/nodes{nb_nodes}/steps{nb_steps}/"
    cache_filename = f"local_gram_DL_diagonalpower{initial_power}"

    if caching:
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

    dl_workload = build_DL_workload(
        matrix, nb_steps=nb_steps, initial_power=initial_power, verbose=verbose
    )  # nT * nT

    # # Old way of doing it.
    # pi = get_pi(nb_nodes=nb_nodes, nb_iterations=nb_steps)  # nT * Tn
    # A = dl_workload @ pi

    pi, pi_inv = get_commutation_reindexing(nb_nodes=nb_nodes, nb_iterations=nb_steps)
    A = dl_workload[
        :, pi_inv
    ]  # Equivalent to the previous A = dl_workload @ pi, see the tests if you want to make sure.

    gram_workload = np.zeros((nb_steps, nb_steps))
    for i in range(nb_nodes):
        if verbose:
            print(f"Computing A{i}")
        Ai = A[:, nb_steps * i : nb_steps * (i + 1)]
        gram_workload += Ai.T @ Ai

    if caching:
        save_to_cache(
            cache_dir=cache_dir,
            filename=cache_filename,
            matrix=gram_workload,
            verbose=verbose,
        )

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
        matrix=matrix,
        nb_steps=nb_steps,
        initial_power=initial_power,
        caching=caching,
        graph_name=graph_name,
        seed=seed,
        verbose=verbose,
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
    if verbose:
        print(f"Cache miss for {cache_path}")
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

    # If post_average==True, we want W in the diagonal since we consider the optimization workload (after averaging), wich corresponds to diag(W) @ \tilde(W) in the paper.
    # If it is false, we instead want \tilde(W) the attacker workload.
    initial_power = int(post_average)

    surrogate_workload = build_local_DL_workload(
        communication_matrix,
        nb_steps=nb_steps,
        initial_power=initial_power,
        graph_name=graph_name,
        seed=seed,
        verbose=verbose,
        caching=caching,
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


def MF_OPTIMAL_DL_variednode(
    communication_matrix,
    nb_nodes,
    nb_steps,
    nb_epochs,
    post_average=True,
    graph_name=None,
    seed=None,
    caching=True,
    verbose=False,
) -> list[np.ndarray]:
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
        C_optimal_allnodes: list(np.array)
    """
    cache_dir = f"cache/correlations_matrixes/{graph_name}/nodes{nb_nodes}/steps{nb_steps}/epochs{nb_epochs}/"

    cache_filename = f"OptimalDLLocal_{'PostAVG' if post_average else "Msg"}"
    if graph_name is None:
        caching = False
    elif graph_require_seed(graph_name):
        if seed is None:
            caching = False
        else:
            cache_filename += f"_seed{seed}"

    if caching:
        cache_results = []
        for node in range(nb_nodes):
            cache_filename_current_node = cache_filename + f"_node{node}"
            cache_result = get_from_cache(
                cache_dir=cache_dir,
                filename=cache_filename_current_node,
                verbose=verbose,
            )
            if cache_result is None:
                cache_results = []
                break
            cache_results.append(cache_result)
        if len(cache_results) == nb_nodes:
            if verbose:
                print(
                    f"Found all cached results under {cache_dir}, skipping computation"
                )
            return cache_results

    # If post_average==True, we want W in the diagonal since we consider the optimization workload (after averaging), wich corresponds to diag(W) @ \tilde(W) in the paper.
    # If it is false, we instead want \tilde(W) the attacker workload.
    initial_power = int(post_average)

    dl_workload = build_DL_workload(
        communication_matrix,
        nb_steps=nb_steps,
        initial_power=initial_power,
        verbose=verbose,
    )  # nT * nT

    # # Old way of doing it.
    # pi = get_pi(nb_nodes=nb_nodes, nb_iterations=nb_steps)  # nT * Tn
    # A = dl_workload @ pi

    pi, pi_inv = get_commutation_reindexing(nb_nodes=nb_nodes, nb_iterations=nb_steps)
    A = dl_workload[
        :, pi_inv
    ]  # Equivalent to the previous A = dl_workload @ pi, see the tests if you want to make sure.

    all_C_optimal = []

    for node in range(nb_nodes):
        Ai = A[:, nb_steps * node : nb_steps * (node + 1)]
        gram_workload = Ai.T @ Ai

        gram_workload_permuted = optimal_factorization._permute_lower_triangle(
            gram_workload
        )
        is_positive_definite = utils.check_positive_definite(gram_workload_permuted)

        if is_positive_definite:
            surrogate_workload = np.linalg.cholesky(gram_workload_permuted)
        else:
            raise NotImplementedError("Non-positive definite Gram workload")

        C_optimal = optimal_factorization.get_optimal_factorization(
            surrogate_workload, nb_steps=nb_steps, nb_epochs=nb_epochs, verbose=verbose
        )
        all_C_optimal.append(C_optimal)

        # Save to cache
        if caching:
            cache_filename_current_node = cache_filename + f"_node{node}"
            save_to_cache(
                cache_dir=cache_dir,
                filename=cache_filename_current_node,
                matrix=C_optimal,
                verbose=verbose,
            )

    return all_C_optimal


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


def BSR_local_factorization(nb_iterations, nb_epochs: Optional[int]):
    """Code inspired from https://github.com/npkalinin/Matrix-Factorization-DP-Training
    If nb_epochs is None, just return the square root, but this may make the sensitivity explode.
    """

    # Workload without momentum
    workload_tensor = torch.ones(nb_iterations)

    # Square root computation
    y = torch.zeros_like(workload_tensor)
    y[0] = torch.sqrt(workload_tensor[0])
    for k in range(1, len(workload_tensor)):
        y[k] = (workload_tensor[k] - torch.dot(y[1:k], y[1:k].flip(0))) / (2 * y[0])

    C = toeplitz(y)
    if nb_epochs is None:
        return C

    band_width = nb_iterations // nb_epochs  # Should force sensitivity of 1

    mask = (
        np.subtract.outer(np.arange(nb_iterations), np.arange(nb_iterations))
        >= band_width
    )
    C[mask] = 0

    # Old code snippet, optimized above
    # C_copy = np.copy(C)
    # for i in range(nb_iterations):
    #     for j in range(nb_iterations):
    #         if i - j >= band_width:
    #             C_copy[i, j] = 0
    # assert np.allclose(C_copy, C)

    return C


def SR_local_factorization(nb_iterations):
    return BSR_local_factorization(nb_iterations=nb_iterations, nb_epochs=None)


def build_projection_workload(
    communication_matrix: np.ndarray, attacker_node: int | list[int], nb_steps: int
) -> np.ndarray:
    """Builds P(attacker_node), the projection workload. Should be used with tilde(W) in the paper, or P @ build_local_dl_workload(.., initial_power=0).

    Args:
        communication_matrix (np.ndarray): The communication matrix
        attacker_node (int or list[int]): The id(s) of the attacking node(s)
        nb_steps (int): Number of steps, to know how many repetitions of the workload will be needed.

    Returns:
        P (np.ndarray): _description_
    """
    n = len(communication_matrix)

    # First, create a projection matrix for a given communication matrix
    projection_lines = []
    if isinstance(attacker_node, int):
        attacker_nodes = [attacker_node]
    else:
        attacker_nodes = attacker_node

    for i in range(n):
        if any(communication_matrix[attacker, i] > 0 for attacker in attacker_nodes):
            projection_line = np.zeros((n))
            projection_line[i] = 1
            projection_lines.append(projection_line)

    projection_matrix = np.array(projection_lines)
    projection_workload = np.kron(np.identity(nb_steps), projection_matrix)
    return projection_workload
