"""
simulator.py
------------
The discrete-event simulator.

Time model
----------
Time advances in integer units. At every unit ``t`` the simulator:

1. **Completes** every process whose finish time equals ``t``
   (its memory is deallocated and merged with neighbouring free blocks).
2. **Arrives** every process whose arrival time equals ``t`` and attempts
   allocation.
3. **Records** a snapshot of the whole system, so the UI can step forwards
   and backwards through history without re-running anything.

Allocation failure policy
-------------------------
``on_failure="reject"`` (default)
    A process that finds no suitable block is rejected immediately. This is
    the classical no-swapping model.

``on_failure="wait"``
    The process stays in a waiting queue and retries at the next time unit
    (the swapping / backing-store model). Such processes appear repeatedly
    in the attempt log, which is why the metrics count *unique* PIDs.

Stepping
--------
The simulator is deliberately stateful so the UI can drive it:
``step()`` advances one time unit, ``run()`` advances to completion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from algorithms import get_algorithm
from analysis.metrics import compute_metrics
from simulation.memory_manager import (
    InitialPartition,
    MemoryBlock,
    MemoryManager,
)
from simulation.process_manager import Process, sort_by_arrival

# Process lifecycle states
PENDING = "Pending"
RUNNING = "Running"
FINISHED = "Finished"
REJECTED = "Rejected"


@dataclass
class Snapshot:
    """Immutable record of the entire system at one time unit."""

    time: int
    blocks: List[MemoryBlock]
    running: List[str]
    pending: List[str]
    finished: List[str]
    rejected: List[str]
    used_memory: int
    free_memory: int
    free_blocks: int
    largest_free_block: int
    events: List[str] = field(default_factory=list)

    @property
    def utilisation(self) -> float:
        total = self.used_memory + self.free_memory
        return round(self.used_memory / total * 100, 2) if total else 0.0


@dataclass
class SimulationResult:
    """Everything the UI needs about one finished simulation."""

    algorithm: str
    metrics: Dict[str, Any]
    records: List[dict]
    history: List[Snapshot]
    final_blocks: List[MemoryBlock]
    events: List[dict]

    def summary_row(self) -> Dict[str, Any]:
        return self.metrics


class Simulator:
    """Drives one workload through one allocation algorithm."""

    def __init__(
        self,
        total_size: int,
        processes: Sequence[Process],
        algorithm: str = "First Fit",
        initial_partitions: Optional[Sequence[InitialPartition]] = None,
        on_failure: str = "reject",
        keep_history: bool = True,
    ) -> None:
        self.total_size = int(total_size)
        self.processes: List[Process] = sort_by_arrival(processes)
        self.algorithm_name = algorithm
        self.initial_partitions = initial_partitions
        self.on_failure = on_failure
        self.keep_history = keep_history

        # A fresh strategy instance per run: Next Fit carries a pointer.
        self.strategy = get_algorithm(algorithm)
        self.memory = MemoryManager(
            total_size=self.total_size,
            initial_partitions=self.initial_partitions,
        )

        self.records: List[dict] = []
        self.history: List[Snapshot] = []
        self.timeline: List[dict] = []
        self.time = 0

        # --- runtime bookkeeping ---------------------------------------
        self._arrivals: Dict[int, List[Process]] = {}
        self._processed_times: set = set()
        self._running: Dict[str, int] = {}      # pid -> finish time
        self._finish_time: Dict[str, int] = {}
        self._status: Dict[str, str] = {}
        self._waiting: List[Process] = []
        self._finished_ids: List[str] = []
        self._rejected_ids: List[str] = []
        self._max_time = 0
        self._initialise()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _initialise(self) -> None:
        self.strategy.reset()
        self.memory.reset()
        self.records = []
        self.history = []
        self.timeline = []
        self.time = 0
        self._running = {}
        self._finish_time = {}
        self._status = {}
        self._waiting = []
        self._finished_ids = []
        self._rejected_ids = []
        self._arrivals = {}
        self._processed_times = set()
        for proc in self.processes:
            self._arrivals.setdefault(proc.arrival, []).append(proc)
            self._status[proc.pid] = PENDING
        if self.processes:
            self._max_time = max(p.finish_if_started_at_arrival
                                 for p in self.processes)
        # No snapshot is taken yet: the first snapshot is produced by the
        # first step() and represents the state *after* time 0 is processed.
        # This keeps exactly one snapshot per time unit.

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def is_finished(self) -> bool:
        """True when no process can still change the memory state.

        A time unit counts as handled only once it has been processed, so a
        workload whose processes all arrive at t = 0 is not declared
        finished before the first step.
        """
        if self._running or self._waiting:
            return False
        return self._all_arrived()

    @property
    def current_snapshot(self) -> Optional[Snapshot]:
        """State at the current point in time (synthesised before the first step)."""
        if self.history:
            return self.history[-1]
        return self._snapshot([])

    def _all_arrived(self) -> bool:
        return all(when in self._processed_times for when in self._arrivals)

    def _status_lists(self) -> Dict[str, List[str]]:
        return {
            RUNNING: [p for p in self._running],
            PENDING: sorted(
                pid for pid, st in self._status.items()
                if st == PENDING
            ),
            FINISHED: list(self._finished_ids),
            REJECTED: list(self._rejected_ids),
        }

    # ------------------------------------------------------------------
    # The event loop
    # ------------------------------------------------------------------
    def step(self) -> Optional[Snapshot]:
        """Process the current time unit and return its snapshot.

        Time is processed as ``t = self.time`` and only advanced *after* the
        unit has been handled, so a process arriving at t = 0 is admitted at
        t = 0 exactly like a real scheduler would.
        """
        if self.is_finished:
            return self.current_snapshot

        t = self.time
        events: List[str] = []

        # 1. completions first - this frees memory for the new arrivals
        for pid, finish in sorted(self._running.items(), key=lambda kv: kv[1]):
            if finish <= t:
                if self.memory.deallocate(pid, time=t):
                    events.append(f"t={t}: {pid} finished, memory freed.")
                self._running.pop(pid, None)
                self._status[pid] = FINISHED
                self._finished_ids.append(pid)

        # 2. new arrivals
        for proc in self._arrivals.get(t, []):
            self._attempt(proc, events)

        # 3. waiting processes retry (only in "wait" mode)
        if self.on_failure == "wait" and self._waiting:
            for proc in list(self._waiting):
                if self._attempt(proc, events):
                    self._waiting.remove(proc)

        # 4. deadlock guard (only reachable in "wait" mode): if nothing is
        # running, nothing else will arrive, and every waiting process just
        # failed, the system can never make progress again. Reject the
        # waiters instead of looping forever.
        if self._waiting and not self._running:
            if not any(when > t for when in self._arrivals):
                for proc in self._waiting:
                    self._status[proc.pid] = REJECTED
                    self._rejected_ids.append(proc.pid)
                self._waiting = []
                events.append(
                    f"t={t}: waiting processes rejected - the system can no "
                    f"longer make progress."
                )

        snapshot = self._snapshot(events)
        if self.keep_history:
            self.history.append(snapshot)
        self._processed_times.add(t)
        self.time = t + 1
        return snapshot

    def _attempt(self, proc: Process, events: List[str]) -> bool:
        """Try to allocate ``proc``; returns True on success."""
        result = self.memory.allocate(
            proc.pid, proc.size, self.strategy, time=self.time
        )
        record = {
            "time": self.time,
            "pid": proc.pid,
            "size": proc.size,
            "duration": proc.duration,
            "arrival": proc.arrival,
            "action": "Allocate",
            "success": result.success,
            "block_start": result.block_start,
            "block_end": result.block_end,
            "search_operations": result.search_operations,
            "algorithm": self.algorithm_name,
            "reason": result.reason,
        }
        self.records.append(record)

        if result.success:
            finish_at = self.time + proc.duration
            self._running[proc.pid] = finish_at
            self._finish_time[proc.pid] = finish_at
            self._status[proc.pid] = RUNNING
            events.append(
                f"t={self.time}: {proc.pid} ({proc.size} KB) allocated at "
                f"{result.block_start}-{result.block_end} KB."
            )
            return True

        if self.on_failure == "wait":
            self._status[proc.pid] = PENDING
            if proc not in self._waiting:
                self._waiting.append(proc)
            events.append(
                f"t={self.time}: {proc.pid} ({proc.size} KB) could not be "
                f"placed - moved to waiting queue."
            )
        else:
            self._status[proc.pid] = REJECTED
            self._rejected_ids.append(proc.pid)
            events.append(
                f"t={self.time}: {proc.pid} ({proc.size} KB) REJECTED - "
                f"{result.reason}"
            )
        return False

    def run(self, max_steps: Optional[int] = None) -> List[Snapshot]:
        """Advance until the simulation finishes (or ``max_steps`` elapse).

        A hard safety cap prevents an infinite loop if a pathological
        workload were ever to keep a process waiting forever.
        """
        limit = max_steps if max_steps is not None else (
            self._max_time + len(self.processes) + 2
        )
        steps = 0
        while not self.is_finished and steps < limit:
            self.step()
            steps += 1
        return self.history

    # ------------------------------------------------------------------
    # Snapshots and results
    # ------------------------------------------------------------------
    def _snapshot(self, events: List[str]) -> Snapshot:
        lists = self._status_lists()
        blocks = self.memory.snapshot()
        if events:
            self.timeline.append({"time": self.time, "events": list(events)})
        return Snapshot(
            time=self.time,
            blocks=blocks,
            running=lists[RUNNING],
            pending=lists[PENDING],
            finished=lists[FINISHED],
            rejected=lists[REJECTED],
            used_memory=self.memory.used_memory(),
            free_memory=self.memory.free_memory(),
            free_blocks=len(self.memory.free_blocks()),
            largest_free_block=self.memory.largest_free_block(),
            events=list(events),
        )

    def push_snapshot(self, events: Optional[List[str]] = None) -> Snapshot:
        """Record the current state without advancing time.

        Used after a manual operation such as compaction, so that the change
        appears in the history and in the memory map like any other event.
        """
        snapshot = self._snapshot(list(events or []))
        if self.keep_history:
            self.history.append(snapshot)
        return snapshot

    def compact(self) -> Dict[str, Any]:
        """Compact memory, recording the layout before and after.

        Returns a description of the change so the UI can display it. Only
        makes sense after the run has finished.
        """
        before_blocks = self.memory.snapshot()
        before_count = len(self.memory.free_blocks())
        relocated = self.memory.compact()
        after_blocks = self.memory.snapshot()
        after_count = len(self.memory.free_blocks())
        self.push_snapshot(
            [f"t={self.time}: memory compacted - {relocated} block(s) moved, "
             f"free blocks {before_count} -> {after_count}."]
        )
        return {
            "relocated_blocks": relocated,
            "free_blocks_before": before_count,
            "free_blocks_after": after_count,
            "free_blocks_reduced": before_count - after_count,
            "largest_before": max(
                (b.size for b in before_blocks if b.is_free), default=0
            ),
            "largest_after": max(
                (b.size for b in after_blocks if b.is_free), default=0
            ),
            "before_blocks": before_blocks,
            "after_blocks": after_blocks,
        }

    def result(self) -> SimulationResult:
        """Package the finished run into a :class:`SimulationResult`."""
        metrics = compute_metrics(
            memory_manager=self.memory,
            records=self.records,
            processes_submitted=len(self.processes),
            algorithm_name=self.algorithm_name,
            makespan=self.time,
        )
        return SimulationResult(
            algorithm=self.algorithm_name,
            metrics=metrics,
            records=list(self.records),
            history=list(self.history),
            final_blocks=self.memory.snapshot(),
            events=list(self.timeline),
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def run_simulation(
    total_size: int,
    processes: Sequence[Process],
    algorithm: str = "First Fit",
    initial_partitions: Optional[Sequence[InitialPartition]] = None,
    on_failure: str = "reject",
    keep_history: bool = True,
) -> SimulationResult:
    """Run one workload to completion with one algorithm (no stepping)."""
    sim = Simulator(
        total_size=total_size,
        processes=processes,
        algorithm=algorithm,
        initial_partitions=initial_partitions,
        on_failure=on_failure,
        keep_history=keep_history,
    )
    sim.run()
    return sim.result()


def run_all_algorithms(
    total_size: int,
    processes: Sequence[Process],
    algorithms: Optional[Sequence[str]] = None,
    initial_partitions: Optional[Sequence[InitialPartition]] = None,
    on_failure: str = "reject",
    keep_history: bool = True,
) -> Dict[str, SimulationResult]:
    """Run the *same* workload with every algorithm.

    The workload is never modified, so any difference between the result
    rows is attributable to the allocation policy alone - which is exactly
    what the comparison table needs to be valid.
    """
    from algorithms import ALGORITHM_ORDER

    names = list(algorithms) if algorithms else list(ALGORITHM_ORDER)
    results: Dict[str, SimulationResult] = {}
    for name in names:
        results[name] = run_simulation(
            total_size=total_size,
            processes=processes,
            algorithm=name,
            initial_partitions=initial_partitions,
            on_failure=on_failure,
            keep_history=keep_history,
        )
    return results
