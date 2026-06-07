"""Path helpers so the simulator is location-independent.

All internal paths are resolved relative to the installed package / repository
via ``__file__``, never via the current working directory or hard-coded user
folders. This lets the simulator run unchanged on any machine (laptop, SLURM
node, ...).
"""

import os


def project_root():
    """Repository root (the directory that contains ``gnss_func`` / ``bayopt``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cacode_dir():
    return os.path.join(project_root(), "CACODE")


def default_results_dir():
    return os.path.join(project_root(), "results")
