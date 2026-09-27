"""
worst_fit.py
------------
Worst Fit: allocate the **largest** free block, leaving the biggest hole
in the remainder.

Behaviour
---------
Scans all free blocks and takes the largest one that can satisfy the
request. The idea is that the leftover of a large block is still useful,
so large processes can still be served later. In practice the leftover
becomes a large hole that is hard to reuse.

Example (total 1000 KB pre-split as 200 | 500 | 300, request 250 KB):
    candidates: 500, 300  ->  picks 500, leaving a 250 KB hole.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from algorithms.base import AllocationStrategy

if TYPE_CHECKING:  # type hints only - avoids a circular import
    from simulation.memory_manager import MemoryBlock


class WorstFit(AllocationStrategy):
    name = "Worst Fit"
    description = (
        "Scans the whole free list and allocates the largest block that can "
        "satisfy the request, leaving a large but hard-to-reuse remainder."
    )
    is_stateful = False

    def select_block(
        self, blocks: List[MemoryBlock], size: int
    ) -> Optional[int]:
        best_index: Optional[int] = None
        best_size: Optional[int] = None

        for index, block in enumerate(blocks):
            if self._fits(block, size):
                if best_size is None or block.size > best_size:
                    best_index = index
                    best_size = block.size

        self.last_examined_count = len(blocks)
        return best_index
