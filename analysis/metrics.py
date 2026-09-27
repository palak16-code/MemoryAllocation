"""
metrics.py
----------
Turns the raw simulation state into the numbers discussed in the report.

All formulas are the standard ones from operating-system literature and are
documented in place so the UI can show the *definition* next to the value.

Definitions used
----------------
Memory utilisation
    ``used_memory / total_memory * 100``

External fragmentation index
    ``1 - (largest free block / total free memory)``

    0%  -> all free memory is usable (e.g. one single hole)
    100% -> free memory is maximally scattered into unusable crumbs.
    Only meaningful when there is at least one free block.

Allocation success rate
    ``processes allocated / processes submitted * 100``

Average search cost
    ``blocks inspected / allocation attempts`` - how much of the block list
    each algorithm had to walk, which is the price paid for the block choice.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Primitive formulas
# ---------------------------------------------------------------------------

def utilisation_percent(used: int, total: int) -> float:
    """Percentage of memory currently occupied."""
    if total <= 0:
        return 0.0
    return round(used / total * 100, 2)


def external_fragmentation_index(free: int, largest: int) -> float:
    """Share of the free memory that is trapped in small unusable holes."""
    if free <= 0:
        return 0.0
    return round((1 - largest / free) * 100, 2)


def average_search_cost(search_ops: int, attempts: int) -> float:
    """Average number of blocks inspected per allocation attempt."""
    if attempts <= 0:
        return 0.0
    return round(search_ops / attempts, 2)


def peak_demand(processes: Iterable[Any]) -> int:
    """Most memory that would be in use at one time if nothing were refused.

    This is a property of the *workload*, not of any algorithm: every
    process is assumed to start the moment it arrives. Comparing it with the
    total memory tells the reader whether the experiment runs under memory
    pressure, which is what makes the success rates interpretable.
    """
    events: List[tuple] = []
    for proc in processes:
        events.append((proc.arrival, proc.size))
        events.append((proc.arrival + proc.duration, -proc.size))

    # Completions (-size) sort before arrivals (+size) at the same instant,
    # which matches the simulator's ordering.
    current = peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta
        peak = max(peak, current)
    return peak


# ---------------------------------------------------------------------------
# Whole-simulation metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    memory_manager: Any,
    records: Sequence[dict],
    processes_submitted: int,
    algorithm_name: str = "",
    makespan: int = 0,
) -> Dict[str, Any]:
    """Build the full metric dictionary for one completed simulation.

    Parameters
    ----------
    memory_manager:
        The final :class:`~simulation.memory_manager.MemoryManager` state.
    records:
        Allocation attempt records (one per attempt, success or failure).
    processes_submitted:
        Number of processes in the workload.
    algorithm_name:
        Label of the algorithm used, copied into the result row.
    makespan:
        Number of time units the simulation ran for.
    """
    attempts = [r for r in records if r.get("action") == "Allocate"]
    successes = [r for r in attempts if r.get("success")]

    # A process that is retried appears several times; count unique PIDs so
    # "processes allocated" means processes, not attempts.
    allocated_pids = {r["pid"] for r in successes}
    rejected_pids = {r["pid"] for r in attempts if not r.get("success")} - allocated_pids

    used = memory_manager.used_memory()
    free = memory_manager.free_memory()
    largest = memory_manager.largest_free_block()
    free_count = len(memory_manager.free_blocks())
    search_ops = memory_manager.search_operations

    return {
        "algorithm": algorithm_name,
        "processes_submitted": processes_submitted,
        "processes_allocated": len(allocated_pids),
        "processes_rejected": len(rejected_pids),
        "allocation_success_rate": round(
            len(allocated_pids) / processes_submitted * 100, 2
        ) if processes_submitted else 0.0,
        "memory_utilisation_pct": utilisation_percent(used, memory_manager.total_size),
        "used_memory_kb": used,
        "total_free_memory_kb": free,
        "free_blocks": free_count,
        "largest_free_block_kb": largest,
        "external_fragmentation_pct": external_fragmentation_index(free, largest),
        "search_operations": search_ops,
        "allocation_attempts": len(attempts),
        "average_search_cost": average_search_cost(search_ops, len(attempts)),
        "makespan": makespan,
    }


def to_comparison_table(results: Iterable[Dict[str, Any]]) -> "Any":
    """Build the algorithm-comparison DataFrame (one row per algorithm)."""
    import pandas as pd  # local import keeps this module importable without pandas

    rows = list(results)
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)

    # Order rows by the canonical algorithm order for stable presentation.
    from algorithms import ALGORITHM_ORDER

    frame["__order"] = frame["algorithm"].apply(
        lambda name: ALGORITHM_ORDER.index(name) if name in ALGORITHM_ORDER else 99
    )
    frame = frame.sort_values("__order").drop(columns="__order")

    pretty = {
        "algorithm": "Algorithm",
        "processes_allocated": "Processes Allocated",
        "processes_rejected": "Processes Rejected",
        "allocation_success_rate": "Success Rate (%)",
        "memory_utilisation_pct": "Memory Utilisation (%)",
        "total_free_memory_kb": "Total Free (KB)",
        "free_blocks": "Free Blocks",
        "largest_free_block_kb": "Largest Free Block (KB)",
        "external_fragmentation_pct": "External Fragmentation (%)",
        "search_operations": "Search Operations",
        "average_search_cost": "Avg Search Cost",
        "processes_submitted": "Processes Submitted",
        "allocation_attempts": "Allocation Attempts",
        "used_memory_kb": "Used Memory (KB)",
        "makespan": "Makespan (time units)",
    }
    return frame.rename(columns=pretty)


#: Short explanations shown as tooltips / captions in the results tab.
METRIC_DEFINITIONS: Dict[str, str] = {
    "Processes Allocated": "Number of distinct processes that received memory.",
    "Processes Rejected": "Processes that never found a block large enough.",
    "Success Rate (%)": "Allocated processes / submitted processes x 100.",
    "Memory Utilisation (%)": "Occupied memory / total memory x 100 at the end of the run.",
    "Total Free (KB)": "Sum of all free blocks at the end of the run.",
    "Free Blocks": "Count of separate free blocks at the end of the run.",
    "Largest Free Block (KB)": "Size of the biggest contiguous free block.",
    "External Fragmentation (%)": "1 - (largest free block / total free memory) x 100.",
    "Search Operations": "Total number of memory blocks inspected across all allocations.",
    "Avg Search Cost": "Search operations / allocation attempts.",
}
