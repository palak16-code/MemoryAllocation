"""
simulation/__init__.py
----------------------
Public surface of the simulation package.
"""

from simulation.memory_manager import (
    AllocationResult,
    InitialPartition,
    MemoryBlock,
    MemoryManager,
)
from simulation.process_manager import (
    Process,
    WorkloadError,
    sort_by_arrival,
    validate_workload,
)
from simulation.simulator import (
    SimulationResult,
    Simulator,
    Snapshot,
    run_all_algorithms,
    run_simulation,
)

__all__ = [
    "AllocationResult",
    "InitialPartition",
    "MemoryBlock",
    "MemoryManager",
    "Process",
    "WorkloadError",
    "sort_by_arrival",
    "validate_workload",
    "SimulationResult",
    "Simulator",
    "Snapshot",
    "run_all_algorithms",
    "run_simulation",
]
