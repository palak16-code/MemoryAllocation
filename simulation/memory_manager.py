"""
memory_manager.py
-----------------
The dynamic-partition memory manager: it owns the physical layout of memory
and performs every state-changing operation (allocate, deallocate, coalesce).

IMPORTANT DESIGN NOTE
---------------------
The memory manager contains NO algorithm logic. It asks an
``AllocationStrategy`` object *which* free block to use, then handles the
bookkeeping itself (splitting, freeing, merging adjacent free blocks).

Consequence: First Fit / Best Fit / Worst Fit / Next Fit provably behave
identically with respect to allocation and deallocation, and differ only in
the *choice* of block. This mirrors how operating-system textbooks separate
the allocation policy from the physical memory manager.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # type hints only - avoids a circular import
    from algorithms.base import AllocationStrategy

FREE = None  # sentinel: a block whose owner is None is free memory


class MemoryError_(RuntimeError):
    """Internal signal that an allocation could not be satisfied."""


@dataclass
class MemoryBlock:
    """A contiguous region of memory.

    start : offset of the first KB of the block (0-based)
    size  : length of the block in KB
    pid   : owning process ID, or ``None`` when the block is free
    """

    start: int
    size: int
    pid: Optional[str] = None

    @property
    def is_free(self) -> bool:
        return self.pid is None

    @property
    def end(self) -> int:
        """Offset one past the last KB (exclusive)."""
        return self.start + self.size

    def label(self) -> str:
        return "FREE" if self.is_free else self.pid

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"MemoryBlock(start={self.start}, size={self.size}, "
            f"pid={self.pid!r})"
        )


@dataclass
class AllocationResult:
    """Outcome of a single allocation attempt (used for the results table)."""

    pid: str
    size: int
    success: bool
    block_start: Optional[int] = None
    block_end: Optional[int] = None
    search_operations: int = 0
    reason: str = ""


@dataclass
class InitialPartition:
    """A user-defined starting block, e.g. an already-busy 200 KB region."""

    size: int
    pid: Optional[str] = None
    label: Optional[str] = None   # cosmetic name for pre-loaded blocks


@dataclass
class MemoryManager:
    """Manages a single contiguous region divided into blocks.

    Parameters
    ----------
    total_size:
        Total memory in KB. Memory is modelled as one *dynamic* region
        (a single free block that is split on demand) because the four
        requested algorithms only apply to dynamic partitioning.
    initial_partitions:
        Optional starting layout, letting the user pre-fragment memory.
    """

    total_size: int
    initial_partitions: Optional[Sequence[InitialPartition]] = None
    blocks: List[MemoryBlock] = field(default_factory=list, init=False)
    search_operations: int = field(default=0, init=False)
    allocation_log: List[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.total_size <= 0:
            raise ValueError("Total memory size must be a positive integer.")
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Return memory to its initial layout and clear counters."""
        self.blocks = self._build_initial_blocks()
        self.search_operations = 0
        self.allocation_log = []

    def _build_initial_blocks(self) -> List[MemoryBlock]:
        if not self.initial_partitions:
            return [MemoryBlock(start=0, size=self.total_size, pid=FREE)]

        blocks: List[MemoryBlock] = []
        cursor = 0
        for part in self.initial_partitions:
            if part.size <= 0:
                raise ValueError("Initial partition sizes must be positive.")
            blocks.append(
                MemoryBlock(start=cursor, size=part.size, pid=part.pid)
            )
            cursor += part.size
        if cursor > self.total_size:
            raise ValueError(
                f"Initial partitions total {cursor} KB but memory is only "
                f"{self.total_size} KB."
            )
        if cursor < self.total_size:
            # Leftover space becomes one trailing free block.
            blocks.append(
                MemoryBlock(start=cursor, size=self.total_size - cursor, pid=FREE)
            )
        return blocks

    # ------------------------------------------------------------------
    # Queries (read-only)
    # ------------------------------------------------------------------
    def free_blocks(self) -> List[MemoryBlock]:
        return [b for b in self.blocks if b.is_free]

    def used_memory(self) -> int:
        return sum(b.size for b in self.blocks if not b.is_free)

    def free_memory(self) -> int:
        return sum(b.size for b in self.blocks if b.is_free)

    def largest_free_block(self) -> int:
        free_sizes = [b.size for b in self.blocks if b.is_free]
        return max(free_sizes) if free_sizes else 0

    def smallest_free_block(self) -> int:
        free_sizes = [b.size for b in self.blocks if b.is_free]
        return min(free_sizes) if free_sizes else 0

    def block_of(self, pid: str) -> Optional[MemoryBlock]:
        for block in self.blocks:
            if block.pid == pid:
                return block
        return None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def allocate(
        self,
        pid: str,
        size: int,
        strategy: AllocationStrategy,
        time: int = 0,
    ) -> AllocationResult:
        """Try to place ``pid`` (needing ``size`` KB) using ``strategy``.

        The strategy only selects a block; all splitting and bookkeeping is
        done here, so every algorithm shares the same allocation mechanics.
        """
        if size <= 0:
            raise ValueError(f"Allocation size for {pid} must be positive.")
        if self.block_of(pid) is not None:
            # Reject duplicate residency: one process owns memory at a time.
            return AllocationResult(
                pid=pid, size=size, success=False,
                reason="Process already holds memory.",
            )

        before = self.search_operations
        chosen_index = strategy.select_block(self.blocks, size)
        self.search_operations += strategy.last_examined_count
        examined = self.search_operations - before

        if chosen_index is None:
            result = AllocationResult(
                pid=pid, size=size, success=False,
                search_operations=examined,
                reason=(
                    f"No free block of {size} KB available "
                    f"(largest free block = {self.largest_free_block()} KB)."
                ),
            )
            self._log(time, result, strategy)
            return result

        block = self.blocks[chosen_index]
        start = block.start

        # --- split / take -------------------------------------------------
        if block.size == size:
            block.pid = pid                       # exact fit: mark whole block
        else:
            leftover = MemoryBlock(
                start=block.start + size, size=block.size - size, pid=FREE
            )
            self.blocks[chosen_index] = MemoryBlock(
                start=block.start, size=size, pid=pid
            )
            self.blocks.insert(chosen_index + 1, leftover)

        result = AllocationResult(
            pid=pid, size=size, success=True,
            block_start=start, block_end=start + size,
            search_operations=examined,
            reason="Allocated.",
        )
        self._log(time, result, strategy)
        return result

    def deallocate(self, pid: str, time: int = 0) -> bool:
        """Free the block owned by ``pid`` and coalesce with free neighbours.

        Returns True if the process actually held memory.
        """
        block = self.block_of(pid)
        if block is None:
            return False
        block.pid = FREE
        self.coalesce()
        self.allocation_log.append(
            {
                "time": time,
                "pid": pid,
                "size": block.size,
                "action": "Deallocate",
                "success": True,
                "block_start": block.start,
                "block_end": block.end,
                "search_operations": 0,
                "algorithm": "",
                "reason": "Memory returned to the free list.",
            }
        )
        return True

    def coalesce(self) -> int:
        """Merge every pair of adjacent free blocks. Returns merges performed.

        Without this step the free-block count would be an artefact of
        process history rather than a property of the current layout.
        """
        merges = 0
        index = 0
        while index < len(self.blocks) - 1:
            current, nxt = self.blocks[index], self.blocks[index + 1]
            if current.is_free and nxt.is_free:
                current.size += nxt.size
                del self.blocks[index + 1]
                merges += 1
            else:
                index += 1
        return merges

    def compact(self) -> int:
        """Move all allocated blocks to the front, leaving one free block.

        This models memory *compaction*: addresses change, but no process
        loses its data. Returns the number of blocks that were relocated.
        """
        if not self.blocks:
            return 0

        relocated = 0
        cursor = 0
        compacted: List[MemoryBlock] = []

        for block in self.blocks:
            if block.is_free:
                continue
            if block.start != cursor:
                relocated += 1
            block.start = cursor
            compacted.append(block)
            cursor += block.size

        free_total = self.total_size - cursor
        if free_total > 0:
            compacted.append(
                MemoryBlock(start=cursor, size=free_total, pid=FREE)
            )

        self.blocks = compacted
        return relocated

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def _log(self, time: int, result: AllocationResult,
             strategy: AllocationStrategy) -> None:
        self.allocation_log.append(
            {
                "time": time,
                "pid": result.pid,
                "size": result.size,
                "action": "Allocate",
                "success": result.success,
                "block_start": result.block_start,
                "block_end": result.block_end,
                "search_operations": result.search_operations,
                "algorithm": strategy.name,
                "reason": result.reason,
            }
        )

    def snapshot(self) -> List[MemoryBlock]:
        """Deep copy of the block list, so history entries never change."""
        return copy.deepcopy(self.blocks)

    def layout_rows(self) -> List[dict]:
        """Flat rows for the memory-map table / visualisation."""
        rows = []
        for order, block in enumerate(self.blocks):
            rows.append(
                {
                    "Order": order,
                    "Start (KB)": block.start,
                    "End (KB)": block.end,
                    "Size (KB)": block.size,
                    "Owner": block.label(),
                    "Type": "Free" if block.is_free else "Process",
                }
            )
        return rows

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return " | ".join(
            f"{b.label()}:{b.size}" for b in self.blocks
        )
