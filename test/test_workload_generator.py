import itertools
import os
import sys
import time

import numpy as np
import pytest

try:
    from simulations.workloads_generator import (
        build_DL_workload,
        build_DL_workload_old,
        get_commutation_matrix,
        get_commutation_reindexing,
    )
except ModuleNotFoundError:
    # Trick to resolve imports
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulations/"))
    )
    from workloads_generator import (
        build_DL_workload,
        build_DL_workload_old,
        get_commutation_matrix,
        get_commutation_reindexing,
    )


def test_get_pi_zero_nodes():
    with pytest.raises(Exception):
        get_commutation_matrix(0, 5)


def test_get_pi_zero_time():
    with pytest.raises(Exception):
        get_commutation_matrix(421, 0)


def test_get_pi_negative_iterations():
    with pytest.raises(Exception):
        get_commutation_matrix(5, -2)


@pytest.mark.parametrize(
    "nb_nodes, nb_iterations",
    [(2, 3), (3, 2), (4, 4), (5, 3), (3, 5), (10, 15), (10, 64)],
)
def test_get_pi_vs_reindexing(nb_nodes, nb_iterations):
    rng = np.random.default_rng()
    p1 = get_commutation_matrix(nb_nodes, nb_iterations)
    p1_inv = np.linalg.inv(p1)

    p2, p2_inv = get_commutation_reindexing(nb_nodes, nb_iterations)
    for _ in range(100):
        W = rng.normal(size=(nb_nodes * nb_iterations, nb_nodes * nb_iterations))

        result1 = p1 @ W
        result2 = W[p2]
        np.testing.assert_allclose(result1, result2, rtol=1e-12, atol=1e-12)

        # Test that p1_inv does the same operation as p2_inv
        result1_inv = p1_inv @ W
        result2_inv = W[p2_inv]
        np.testing.assert_allclose(result1_inv, result2_inv, rtol=1e-12, atol=1e-12)

        # Test that p1 and p1_inv are inverses
        identity1 = p1 @ p1_inv
        np.testing.assert_allclose(
            identity1, np.eye(nb_nodes * nb_iterations), rtol=1e-12, atol=1e-12
        )

        # Test that p2 and p2_inv are inverses
        # Compose the permutations and check if it gives the identity permutation
        composed = p2[p2_inv]
        np.testing.assert_array_equal(composed, np.arange(0, nb_nodes * nb_iterations))

        result1 = W @ p1
        result2 = W[:, p2_inv]
        np.testing.assert_allclose(result1, result2, rtol=1e-12, atol=1e-12)

        result1 = W @ p1_inv
        result2 = W[:, p2]
        np.testing.assert_allclose(result1, result2, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "n, nb_steps, initial_power",
    [
        (2, 3, 0),
        (2, 3, 1),
        (3, 2, 0),
        (3, 2, 1),
        (4, 4, 0),
        (4, 4, 1),
        (5, 3, 0),
        (5, 3, 1),
        (3, 5, 0),
        (3, 5, 1),
        (10, 15, 0),
        (10, 15, 1),
        (10, 64, 0),
        (10, 64, 1),
        (1, 1, 0),
        (1, 1, 1),
        (6, 7, 0),
        (6, 7, 1),
        (8, 9, 0),
        (8, 9, 1),
        (12, 5, 0),
        (12, 5, 1),
        (15, 3, 0),
        (15, 3, 1),
        (20, 2, 0),
        (20, 2, 1),
    ],
)
def test_equivalence_and_benchmark(n, nb_steps, initial_power):
    matrix = np.random.rand(n, n)
    matrix = (matrix + matrix.T) / 2  # Make symmetric for realism

    # Test equivalence
    result1 = build_DL_workload(matrix, nb_steps, initial_power)
    result2 = build_DL_workload_old(matrix, nb_steps, initial_power)
    assert np.allclose(result1, result2), "Outputs of both functions differ!"


@pytest.mark.parametrize(
    "n, nb_steps, initial_power",
    [
        (2, 2, 0),
        (2, 2, 1),
        (3, 3, 0),
        (3, 3, 1),
        (5, 5, 0),
        (5, 5, 1),
        (7, 4, 0),
        (7, 4, 1),
        (9, 6, 0),
        (9, 6, 1),
    ],
)
def test_workload_output_shape(n, nb_steps, initial_power):
    matrix = np.random.rand(n, n)
    matrix = (matrix + matrix.T) / 2
    result = build_DL_workload(matrix, nb_steps, initial_power)
    assert result.shape[0] == nb_steps * n, "Output shape mismatch in steps"
    assert result.shape[1] == n * nb_steps, "Output shape mismatch in nodes"


@pytest.mark.parametrize(
    "n, nb_steps, initial_power",
    [
        (2, 2, 0),
        (2, 2, 1),
        (3, 3, 0),
        (3, 3, 1),
        (5, 5, 0),
        (5, 5, 1),
    ],
)
def test_workload_non_negative(n, nb_steps, initial_power):
    matrix = np.abs(np.random.rand(n, n))
    matrix = (matrix + matrix.T) / 2
    result = build_DL_workload(matrix, nb_steps, initial_power)
    assert np.all(result >= 0), "Workload contains negative values"


def test_workload_zero_matrix():
    n, nb_steps, initial_power = 4, 4, 1
    matrix = np.zeros((n, n))
    result = build_DL_workload(matrix, nb_steps, initial_power)
    assert np.allclose(result, 0), "Workload should be all zeros for zero matrix"
