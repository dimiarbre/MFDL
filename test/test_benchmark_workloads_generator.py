import os
import sys

import numpy as np
import pytest

try:
    from simulations import utils as sim_utils
    from simulations.workloads_generator import (
        MF_ANTIPGD,
        MF_LDP,
        MF_OPTIMAL_DL,
        BSR_local_factorization,
        MF_OPTIMAL_DL_variednode,
        MF_OPTIMAL_local,
        build_DL_workload,
        build_DL_workload_old,
        build_participation_matrix,
        get_commutation_matrix,
    )
except:
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../simulations/"))
    )
    import utils as sim_utils
    from workloads_generator import (
        MF_ANTIPGD,
        MF_LDP,
        MF_OPTIMAL_DL,
        BSR_local_factorization,
        MF_OPTIMAL_DL_variednode,
        MF_OPTIMAL_local,
        build_DL_workload,
        build_DL_workload_old,
        build_participation_matrix,
        get_commutation_matrix,
    )

# Require pytest-benchmark plugin for these benchmarks; skip tests if it's not available.
pytest.importorskip("pytest_benchmark")


@pytest.mark.benchmark(group="get_pi")
def test_get_pi_speed(benchmark):
    benchmark(get_commutation_matrix, 100, 100)


@pytest.mark.parametrize(
    "num_epochs, participation_interval",
    [
        (100, 10),
        (120, 20),
        (240, 30),
        (300, 50),
        (480, 60),
    ],
)
@pytest.mark.benchmark(group="build_participation_matrix")
def test_build_participation_matrix_speed(
    benchmark, num_epochs, participation_interval
):
    benchmark(build_participation_matrix, num_epochs, participation_interval)


@pytest.fixture
def random_matrix():
    np.random.seed(42)
    n = 20
    nb_steps = 100
    initial_power = 1
    matrix = np.random.rand(n, n)
    matrix = (matrix + matrix.T) / 2  # Make symmetric
    return matrix, nb_steps, initial_power


@pytest.mark.benchmark(group="dl_workload")
def test_build_DL_workload_speed(benchmark, random_matrix):
    matrix, nb_steps, initial_power = random_matrix
    benchmark(build_DL_workload, matrix, nb_steps, initial_power)


@pytest.mark.benchmark(group="dl_workload")
def test_build_DL_workload_old_speed(benchmark, random_matrix):
    matrix, nb_steps, initial_power = random_matrix
    benchmark(build_DL_workload_old, matrix, nb_steps, initial_power)


@pytest.mark.benchmark(group="factorization_workloads")
@pytest.mark.parametrize(
    "method_name",
    [
        "Unnoised baseline",
        "LDP",
        "ANTIPGD",
        "BSR_BANDED_LOCAL",
        "OPTIMAL_LOCAL",
        "OPTIMAL_DL_MSG",
        "OPTIMAL_DL_POSTAVG",
        "OPTIMAL_DL_LOCALCOR",
    ],
)
def test_factorization_workloads_speed(benchmark, method_name):
    """Benchmark the time to compute workload/factorization matrices for each method.

    Uses small sizes so the test stays fast while still measuring relative costs.
    """
    # Small configuration for a fast benchmark
    n = 6
    nb_epochs = 10
    nb_batches = 5
    seed = 421

    num_steps = nb_epochs * nb_batches

    # Build a communication matrix via a small graph
    G = sim_utils.get_graph("expander", n, seed=seed)
    comm_matrix = sim_utils.get_communication_matrix(G)

    def make_config():
        if method_name == "Unnoised baseline":
            return np.zeros((num_steps, num_steps))
        if method_name == "LDP":
            return MF_LDP(nb_nodes=1, nb_iterations=num_steps)
        if method_name == "ANTIPGD":
            return MF_ANTIPGD(nb_nodes=1, nb_iterations=num_steps)
        if method_name == "BSR_BANDED_LOCAL":
            return BSR_local_factorization(nb_iterations=num_steps, nb_epochs=nb_epochs)
        if method_name == "OPTIMAL_LOCAL":
            return MF_OPTIMAL_local(
                communication_matrix=comm_matrix,
                nb_nodes=n,
                nb_steps=num_steps,
                nb_epochs=nb_epochs,
                caching=False,
                verbose=False,
            )
        if method_name == "OPTIMAL_DL_MSG":
            return MF_OPTIMAL_DL(
                communication_matrix=comm_matrix,
                nb_nodes=n,
                nb_steps=num_steps,
                nb_epochs=nb_epochs,
                post_average=False,
                graph_name="expander",
                seed=seed,
                caching=False,
                verbose=False,
            )
        if method_name == "OPTIMAL_DL_POSTAVG":
            return MF_OPTIMAL_DL(
                communication_matrix=comm_matrix,
                nb_nodes=n,
                nb_steps=num_steps,
                nb_epochs=nb_epochs,
                post_average=True,
                graph_name="expander",
                seed=seed,
                caching=False,
                verbose=False,
            )
        if method_name == "OPTIMAL_DL_LOCALCOR":
            return MF_OPTIMAL_DL_variednode(
                communication_matrix=comm_matrix,
                nb_nodes=n,
                nb_steps=num_steps,
                nb_epochs=nb_epochs,
                post_average=True,
                graph_name="expander",
                seed=seed,
                caching=False,
                verbose=False,
            )
        raise ValueError(f"Unknown method {method_name}")

    # benchmark the generator call
    result = benchmark(make_config)
    # Basic sanity checks on result
    assert result is not None
