import itertools
import os
import sys

import numpy as np
import pytest

try:
    from simulations.workloads_generator import get_pi, get_pi_reindexing
except ModuleNotFoundError:
    # Trick to resolve imports
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulations/"))
    )
    from workloads_generator import get_pi, get_pi_reindexing


def test_get_pi_zero_nodes():
    with pytest.raises(Exception):
        get_pi(0, 5)


def test_get_pi_zero_time():
    with pytest.raises(Exception):
        get_pi(421, 0)


def test_get_pi_negative_iterations():
    with pytest.raises(Exception):
        get_pi(5, -2)


@pytest.mark.parametrize(
    "nb_nodes, nb_iterations",
    [(2, 3), (3, 2), (4, 4), (5, 3), (3, 5), (10, 15), (10, 64)],
)
def test_get_pi_vs_reindexing(nb_nodes, nb_iterations):
    rng = np.random.default_rng()
    p1 = get_pi(nb_nodes, nb_iterations)
    p1_inv = np.linalg.inv(p1)

    p2, p2_inv = get_pi_reindexing(nb_nodes, nb_iterations)
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
