import itertools

import numpy as np
import pytest

from simulations.workloads_generator import get_pi


def test_get_pi_zero_nodes():
    with pytest.raises(Exception):
        get_pi(0, 5)


def test_get_pi_zero_time():
    with pytest.raises(Exception):
        get_pi(421, 0)


def test_get_pi_negative_iterations():
    with pytest.raises(Exception):
        get_pi(5, -2)
