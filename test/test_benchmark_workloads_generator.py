import pytest

from simulations.workloads_generator import get_pi


@pytest.mark.benchmark(group="get_pi")
def test_get_pi_speed(benchmark):
    benchmark(get_pi, 100, 100)
