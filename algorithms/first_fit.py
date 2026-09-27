"""
first_fit.py
------------
First Fit: allocate the **first** (lowest-address) free block large enough.

Behaviour
---------
Scans blocks from address 0 upward and takes the first hole that fits.
Simple and fast, but tends to leave small unusable holes scattered low in
memory - the classic cause of external fragmentation.

Example (total 1000 KB, one free block, request 300 KB):
    [ FREE 1000 ]  ->  [ P1 300 ][ FREE 700 ]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from algorithms.base import AllocationStrategy

if TYPE_CHECKING:  # type hints only - avoids a circular import
    from simulation.memory_manager import MemoryBlock


class FirstFit(AllocationStrategy):
    name = "First Fit"
    description = (
        "Scans memory from the lowest address upward and allocates the "
        "first free block that is large enough. Fast, but leaves small "
        "unusable holes near the beginning of memory."
    )
    is_stateful = False

    def select_block(
        self, blocks: List[MemoryBlock], size: int
    ) -> Optional[int]:
        for index, block in enumerate(blocks):
            if self._fits(block, size):
                # index + 1 blocks inspected (0-based index + itself).
                self.last_examined_count = index + 1
                return index

        # No block fitted: every block had to be inspected.
        self.last_examined_count = len(blocks)
        return None
