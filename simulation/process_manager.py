"""
process_manager.py
------------------
Defines the *Process* data model and helpers for validating a workload.

A process is described by four values only (as required by the project brief):

    pid            unique identifier, e.g. "P1"
    size           memory requirement in KB
    arrival        arrival time in time units
    duration       how long the process runs before terminating

Nothing in this module knows about memory or algorithms. It is pure data
plus input validation, which keeps it trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Iterable, List, Sequence


class WorkloadError(ValueError):
    """Raised when a user-supplied workload is invalid."""


@dataclass(frozen=True)
class Process:
    """One unit of work entering the system.

    Immutable (``frozen=True``) so that simulation history snapshots can never
    be corrupted by later edits in the UI.
    """

    pid: str
    size: int          # memory requirement, KB
    arrival: int       # arrival time, time units
    duration: int      # execution length, time units

    def __post_init__(self) -> None:
        if not str(self.pid).strip():
            raise WorkloadError("Process ID cannot be empty.")
        if self.size <= 0:
            raise WorkloadError(
                f"Process {self.pid}: memory size must be a positive integer."
            )
        if self.arrival < 0:
            raise WorkloadError(
                f"Process {self.pid}: arrival time cannot be negative."
            )
        if self.duration <= 0:
            raise WorkloadError(
                f"Process {self.pid}: duration must be a positive integer."
            )

    # -- convenience ------------------------------------------------------
    @property
    def finish_if_started_at_arrival(self) -> int:
        """Time at which the process would finish if it never waited."""
        return self.arrival + self.duration

    def with_pid(self, pid: str) -> "Process":
        """Return a copy carrying a different ID (used by the generator)."""
        return replace(self, pid=pid)

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "size": self.size,
            "arrival": self.arrival,
            "duration": self.duration,
        }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_workload(processes: Sequence[Process]) -> List[str]:
    """Return a list of human-readable problems (empty list == workload OK).

    Used by the UI to show friendly errors instead of raising deep in the
    simulator, and by tests as a single assertion point.
    """
    problems: List[str] = []

    if not processes:
        return ["No processes defined. Add at least one process to simulate."]

    seen: dict[str, int] = {}
    for index, proc in enumerate(processes):
        if proc.pid in seen:
            problems.append(
                f"Duplicate process ID '{proc.pid}' "
                f"(rows {seen[proc.pid] + 1} and {index + 1})."
            )
        else:
            seen[proc.pid] = index

        if proc.size <= 0:
            problems.append(f"Process {proc.pid}: size must be > 0 (got {proc.size}).")
        if proc.arrival < 0:
            problems.append(
                f"Process {proc.pid}: arrival time must be >= 0 (got {proc.arrival})."
            )
        if proc.duration <= 0:
            problems.append(
                f"Process {proc.pid}: duration must be > 0 (got {proc.duration})."
            )

    return problems


def sort_by_arrival(processes: Iterable[Process]) -> List[Process]:
    """Arrival order, ties broken by ID so runs are deterministic."""
    return sorted(processes, key=lambda p: (p.arrival, p.pid))


# ---------------------------------------------------------------------------
# Random workload generation (experiment support)
# ---------------------------------------------------------------------------

#: Ready-made workload sizes offered in the UI: (label, process count).
WORKLOAD_SIZES = {
    "Small": 8,
    "Medium": 20,
    "Large": 40,
}


def generate_workload(
    count: int,
    min_size: int,
    max_size: int,
    max_arrival: int = 5,
    min_duration: int = 1,
    max_duration: int = 6,
    seed: int | None = None,
) -> List[Process]:
    """Build a reproducible random workload.

    A fixed ``seed`` gives the same workload every time, which is what makes
    an algorithm comparison a fair experiment: the identical input is replayed
    for each algorithm.
    """
    if count <= 0:
        raise WorkloadError("Workload size must be at least 1 process.")
    if min_size <= 0 or max_size < min_size:
        raise WorkloadError(
            "Process size range is invalid: need 0 < min size <= max size."
        )
    if max_arrival < 0:
        raise WorkloadError("Maximum arrival time cannot be negative.")
    if min_duration <= 0 or max_duration < min_duration:
        raise WorkloadError(
            "Duration range is invalid: need 0 < min duration <= max duration."
        )

    rng = random.Random(seed)
    processes: List[Process] = []
    for index in range(1, count + 1):
        processes.append(
            Process(
                pid=f"P{index}",
                size=rng.randint(min_size, max_size),
                arrival=rng.randint(0, max_arrival),
                duration=rng.randint(min_duration, max_duration),
            )
        )
    return processes
