"""
best_fit.py
-----------
Best Fit: allocate the **smallest** free block that is large enough.

Behaviour
---------
Scans all free blocks and keeps the one whose size is closest to (but not
below) the request. Aimed at keeping large blocks free for future large
processes, at the cost of a full scan of the free list on every allocation.

Example (total 1000 KB pre-split as 200 | 500 | 300, request 250 KB):
    candidates: 200 (too small), 500 (fits), 300 (fits)  ->  picks 300.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from algorithms.base import AllocationStrategy

if TYPE_CHECKING:  # type hints only - avoids a circular import
    from simulation.memory_manager import MemoryBlock


class BestFit(AllocationStrategy):
    name = "Best Fit"
    description = (
        "Scans the whole free list and allocates the smallest block that can "
        "still satisfy the request. Tries to keep large blocks free, but "
        "leaves many tiny unusable holes behind."
    )
    is_stateful = False

    def select_block(
        self, blocks: List[MemoryBlock], size: int
    ) -> Optional[int]:
        best_index: Optional[int] = None
        best_size: Optional[int] = None

        # Best Fit must always examine every block: a later, smaller hole
        # may still be a better choice.
        for index, block in enumerate(blocks):
            if self._fits(block, size):
                if best_size is None or block.size < best_size:
                    best_index = index
                    best_size = block.size

        self.last_examined_count = len(blocks)
        return best_index
