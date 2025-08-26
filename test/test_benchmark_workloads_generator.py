import pytest

from simulations.workloads_generator import (
    build_participation_matrix,
    build_participation_matrix_v2,
    get_pi,
)


@pytest.mark.benchmark(group="get_pi")
def test_get_pi_speed(benchmark):
    benchmark(get_pi, 100, 100)


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
