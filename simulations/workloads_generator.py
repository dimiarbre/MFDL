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


def MF_LDP(nb_nodes, nb_iterations):
    C = np.identity(nb_nodes * nb_iterations)
    return C
