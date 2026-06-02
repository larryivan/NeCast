from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


class SlurmEnvironmentError(ValueError):
    """Raised when SLURM topology variables are present but inconsistent."""


@dataclass(frozen=True)
class SlurmArrayContext:
    """A regular SLURM job-array allocation expressed as a zero-based rank."""

    task_id: int
    rank: int
    count: int
    minimum: int
    maximum: int
    step: int


@dataclass(frozen=True)
class DistributedContext:
    """Global process topology derived from SLURM or explicit overrides."""

    rank: int
    world_size: int
    task_rank: int
    task_count: int
    array: SlurmArrayContext | None
    source: str

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _parse_int(
    environ: Mapping[str, str],
    name: str,
    *,
    minimum: int,
) -> int | None:
    raw = environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(str(raw), 10)
    except ValueError as exc:
        raise SlurmEnvironmentError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        comparator = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise SlurmEnvironmentError(f"{name} must be {comparator}, got {value}")
    return value


def slurm_cpu_count(environ: Mapping[str, str] | None = None) -> int | None:
    """Return the CPUs assigned to this SLURM task, when explicitly available."""

    env = _environment(environ)
    return _parse_int(env, "SLURM_CPUS_PER_TASK", minimum=1)


def allocated_cpu_count(
    environ: Mapping[str, str] | None = None,
    *,
    override: int | None = None,
) -> int:
    """Resolve the CPU budget visible to one local process.

    Explicit overrides take precedence over SLURM. Outside SLURM, process CPU
    affinity is preferred over the machine-wide CPU count so a cgroup or taskset
    allocation is not accidentally oversubscribed.
    """

    if override is not None:
        if isinstance(override, bool) or not isinstance(override, int) or override <= 0:
            raise ValueError("CPU-count override must be a positive integer")
        return override

    slurm_count = slurm_cpu_count(environ)
    if slurm_count is not None:
        return slurm_count

    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            affinity_count = len(get_affinity(0))
        except OSError:
            affinity_count = 0
        if affinity_count > 0:
            return affinity_count
    return max(1, os.cpu_count() or 1)


def slurm_array_context(
    environ: Mapping[str, str] | None = None,
) -> SlurmArrayContext | None:
    """Parse a regular SLURM array into a compact zero-based rank.

    SLURM does not expose an ordinal for arbitrary sparse array specifications.
    Such layouts are rejected instead of silently assigning overlapping work.
    Contiguous and regularly stepped arrays (for example ``0-19`` or
    ``2-20:2``) are supported.
    """

    env = _environment(environ)
    task_id = _parse_int(env, "SLURM_ARRAY_TASK_ID", minimum=0)
    if task_id is None:
        return None

    minimum = _parse_int(env, "SLURM_ARRAY_TASK_MIN", minimum=0)
    maximum = _parse_int(env, "SLURM_ARRAY_TASK_MAX", minimum=0)
    step = _parse_int(env, "SLURM_ARRAY_TASK_STEP", minimum=1)
    count = _parse_int(env, "SLURM_ARRAY_TASK_COUNT", minimum=1)
    if minimum is None or maximum is None or count is None:
        raise SlurmEnvironmentError(
            "SLURM array tasks require SLURM_ARRAY_TASK_MIN, "
            "SLURM_ARRAY_TASK_MAX, and SLURM_ARRAY_TASK_COUNT"
        )
    if step is None:
        step = 1
    if maximum < minimum:
        raise SlurmEnvironmentError(
            "SLURM_ARRAY_TASK_MAX must not be smaller than SLURM_ARRAY_TASK_MIN"
        )
    if task_id < minimum or task_id > maximum:
        raise SlurmEnvironmentError(
            "SLURM_ARRAY_TASK_ID lies outside the declared array range"
        )
    delta = task_id - minimum
    if delta % step != 0:
        raise SlurmEnvironmentError(
            "SLURM_ARRAY_TASK_ID is not aligned to SLURM_ARRAY_TASK_STEP"
        )

    expected_count = (maximum - minimum) // step + 1
    rank = delta // step
    if expected_count != count or rank >= count:
        raise SlurmEnvironmentError(
            "SLURM array metadata is not a regular range; use explicit rank/world overrides"
        )
    return SlurmArrayContext(
        task_id=task_id,
        rank=rank,
        count=count,
        minimum=minimum,
        maximum=maximum,
        step=step,
    )


def distributed_context(
    environ: Mapping[str, str] | None = None,
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> DistributedContext:
    """Resolve one global rank from SLURM tasks and an optional job array.

    When both mechanisms are present, process ranks are nested inside array
    ranks: ``global_rank = array_rank * SLURM_NTASKS + SLURM_PROCID``.
    Passing ``rank`` and ``world_size`` explicitly bypasses SLURM discovery.
    """

    if (rank is None) != (world_size is None):
        raise ValueError("rank and world_size overrides must be provided together")
    if rank is not None and world_size is not None:
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size <= 0
            or rank < 0
            or rank >= world_size
        ):
            raise ValueError("distributed overrides require 0 <= rank < world_size")
        return DistributedContext(
            rank=rank,
            world_size=world_size,
            task_rank=rank,
            task_count=world_size,
            array=None,
            source="explicit",
        )

    env = _environment(environ)
    task_rank = _parse_int(env, "SLURM_PROCID", minimum=0)
    task_count = _parse_int(env, "SLURM_NTASKS", minimum=1)
    if task_rank is not None and task_count is None:
        raise SlurmEnvironmentError("SLURM_PROCID is set without SLURM_NTASKS")
    if task_count is None:
        task_count = 1
    if task_rank is None:
        if task_count > 1:
            raise SlurmEnvironmentError(
                "SLURM_NTASKS is greater than one but SLURM_PROCID is unavailable; "
                "launch through srun or provide explicit rank/world overrides"
            )
        task_rank = 0
    if task_rank >= task_count:
        raise SlurmEnvironmentError("SLURM_PROCID must be smaller than SLURM_NTASKS")

    array = slurm_array_context(env)
    if array is None:
        return DistributedContext(
            rank=task_rank,
            world_size=task_count,
            task_rank=task_rank,
            task_count=task_count,
            array=None,
            source="slurm" if task_count > 1 else "single",
        )

    return DistributedContext(
        rank=array.rank * task_count + task_rank,
        world_size=array.count * task_count,
        task_rank=task_rank,
        task_count=task_count,
        array=array,
        source="slurm_array",
    )
