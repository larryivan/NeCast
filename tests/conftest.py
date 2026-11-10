import os

import pytest


@pytest.fixture(autouse=True)
def outside_slurm(monkeypatch):
    """The simulator reads SLURM_* variables, so tests run inside a Slurm job start without them."""
    for name in list(os.environ):
        if name.startswith("SLURM_"):
            monkeypatch.delenv(name)
