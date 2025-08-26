import numpy as np


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
