"""
next_fit.py
-----------
Next Fit: like First Fit, but the search resumes from **where the previous
allocation ended** instead of from the start of memory.

Behaviour
---------
A "roving pointer" (``_next_address``) remembers the address just after the
last block handed out. The next allocation scans from there to the end of
memory and, if nothing fits, wraps around and scans from address 0. This is
the closest of the four algorithms to a real allocator, because rescanning
the whole free list on every allocation is expensive.

Why the pointer stores an *address* and not a block index
---------------------------------------------------------
A block index is not stable: allocating splits a block and inserts a new one
into the list, and deallocating merges blocks and deletes entries. An address
is stable, so the pointer always refers to the same place in the address
space no matter how the list is restructured.

Statefulness
------------
Unlike the other three, Next Fit is *stateful*: the same block list can yield
different results depending on history. :meth:`reset` is called by the
simulator before every run.

Example (total 1000 KB, then a 100 KB request after the pointer has moved):
    P1:300 | P2:200 | P3:300 | FREE:200   pointer at 800
    free P1  ->  FREE:300 | P2:200 | P3:300 | FREE:200
    request 100 KB  ->  Next Fit uses the 200 KB hole at 800, because the
                        300 KB hole at address 0 lies *behind* the pointer.
                        First Fit would have used address 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from algorithms.base import AllocationStrategy

if TYPE_CHECKING:  # type hints only - avoids a circular import
    from simulation.memory_manager import MemoryBlock


class NextFit(AllocationStrategy):
    name = "Next Fit"
    description = (
        "Identical to First Fit except the search continues from the address "
        "after the previous allocation, wrapping round when it reaches the "
        "end. Cheaper search, but holes behind the pointer are ignored. "
        "Stateful: results depend on allocation order."
    )
    is_stateful = True

    def __init__(self) -> None:
        super().__init__()
        self._next_address: int = 0

    def reset(self) -> None:
        super().reset()
        self._next_address = 0

    @property
    def next_address(self) -> int:
        """Address the next search will start from (exposed for teaching)."""
        return self._next_address

    def select_block(
        self, blocks: List[MemoryBlock], size: int
    ) -> Optional[int]:
        if not blocks:
            self.last_examined_count = 0
            return None

        examined = 0
        chosen: Optional[int] = None

        # --- pass 1: from the roving pointer to the end of memory --------
        for index, block in enumerate(blocks):
            if block.end > self._next_address:
                examined += 1
                if self._fits(block, size):
                    chosen = index
                    break

        # --- pass 2: wrap around and scan the low addresses -------------
        if chosen is None:
            for index, block in enumerate(blocks):
                if block.end <= self._next_address:
                    examined += 1
                    if self._fits(block, size):
                        chosen = index
                        break

        self.last_examined_count = examined

        if chosen is not None:
            # Resume the search just after the portion we handed out. Using
            # `start + size` (not `block.end`) matters: the leftover of a
            # split block is still free, but the pointer must not skip it.
            self._next_address = blocks[chosen].start + size

        # On failure the pointer is deliberately left untouched: a refused
        # allocation does not move the roving pointer in a real allocator.
        return chosen
