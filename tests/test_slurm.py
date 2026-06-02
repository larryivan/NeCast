from __future__ import annotations

import pytest

from necast.simulator.slurm import (
    SlurmEnvironmentError,
    allocated_cpu_count,
    distributed_context,
    slurm_array_context,
    slurm_cpu_count,
)
from necast.simulator.utils import resolve_workers


def test_slurm_cpu_allocation_and_explicit_override() -> None:
    env = {"SLURM_CPUS_PER_TASK": "16"}

    assert slurm_cpu_count(env) == 16
    assert allocated_cpu_count(env) == 16
    assert allocated_cpu_count(env, override=7) == 7
    assert resolve_workers(-1, environ=env) == 15
    assert resolve_workers(-1, environ=env, cpu_count=5) == 4
    assert resolve_workers(3, environ=env) == 3


@pytest.mark.parametrize("value", ["0", "-1", "many", "1.5"])
def test_invalid_slurm_cpu_allocation_is_rejected(value: str) -> None:
    with pytest.raises(SlurmEnvironmentError, match="SLURM_CPUS_PER_TASK"):
        slurm_cpu_count({"SLURM_CPUS_PER_TASK": value})


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_cpu_override_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="CPU-count override"):
        allocated_cpu_count({}, override=value)


def test_regular_slurm_array_context() -> None:
    context = slurm_array_context(
        {
            "SLURM_ARRAY_TASK_ID": "8",
            "SLURM_ARRAY_TASK_MIN": "2",
            "SLURM_ARRAY_TASK_MAX": "20",
            "SLURM_ARRAY_TASK_STEP": "2",
            "SLURM_ARRAY_TASK_COUNT": "10",
        }
    )

    assert context is not None
    assert context.task_id == 8
    assert context.rank == 3
    assert context.count == 10
    assert context.minimum == 2
    assert context.maximum == 20
    assert context.step == 2


def test_incomplete_or_sparse_slurm_array_is_rejected() -> None:
    with pytest.raises(SlurmEnvironmentError, match="require"):
        slurm_array_context({"SLURM_ARRAY_TASK_ID": "1"})

    with pytest.raises(SlurmEnvironmentError, match="regular range"):
        slurm_array_context(
            {
                "SLURM_ARRAY_TASK_ID": "7",
                "SLURM_ARRAY_TASK_MIN": "1",
                "SLURM_ARRAY_TASK_MAX": "7",
                "SLURM_ARRAY_TASK_STEP": "1",
                "SLURM_ARRAY_TASK_COUNT": "3",
            }
        )


def test_distributed_context_combines_array_and_process_ranks() -> None:
    context = distributed_context(
        {
            "SLURM_PROCID": "2",
            "SLURM_NTASKS": "4",
            "SLURM_ARRAY_TASK_ID": "3",
            "SLURM_ARRAY_TASK_MIN": "0",
            "SLURM_ARRAY_TASK_MAX": "7",
            "SLURM_ARRAY_TASK_STEP": "1",
            "SLURM_ARRAY_TASK_COUNT": "8",
        }
    )

    assert context.rank == 14
    assert context.world_size == 32
    assert context.task_rank == 2
    assert context.task_count == 4
    assert context.array is not None
    assert context.array.rank == 3
    assert context.source == "slurm_array"
    assert context.distributed


def test_array_only_context_and_single_process_default() -> None:
    array_context = distributed_context(
        {
            "SLURM_ARRAY_TASK_ID": "4",
            "SLURM_ARRAY_TASK_MIN": "0",
            "SLURM_ARRAY_TASK_MAX": "9",
            "SLURM_ARRAY_TASK_COUNT": "10",
        }
    )
    assert (array_context.rank, array_context.world_size) == (4, 10)

    single = distributed_context({})
    assert (single.rank, single.world_size) == (0, 1)
    assert not single.distributed
    assert single.source == "single"


def test_explicit_distributed_override_takes_precedence() -> None:
    context = distributed_context(
        {"SLURM_PROCID": "invalid", "SLURM_NTASKS": "invalid"},
        rank=3,
        world_size=5,
    )

    assert (context.rank, context.world_size) == (3, 5)
    assert context.source == "explicit"


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"SLURM_PROCID": "1"}, "without SLURM_NTASKS"),
        ({"SLURM_NTASKS": "2"}, "SLURM_PROCID is unavailable"),
        ({"SLURM_PROCID": "2", "SLURM_NTASKS": "2"}, "must be smaller"),
    ],
)
def test_inconsistent_slurm_process_topology_is_rejected(
    env: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(SlurmEnvironmentError, match=message):
        distributed_context(env)


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [(None, 2), (1, None), (-1, 2), (2, 2), (0, 0), (True, 2)],
)
def test_invalid_explicit_distributed_override_is_rejected(
    rank,
    world_size,
) -> None:
    with pytest.raises(ValueError):
        distributed_context({}, rank=rank, world_size=world_size)
