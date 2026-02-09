"""
Simulations module for MFDL.
Contains workload generators and simulation utilities for decentralized learning.
"""

from .workloads_generator import build_DL_workload, build_projection_workload

__all__ = [
    "build_DL_workload",
    "build_projection_workload",
]
