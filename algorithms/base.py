"""
base.py
-------
Defines the *strategy interface* shared by all four allocation algorithms.

An algorithm's only job is: **given the current list of memory blocks and a
requested size, decide which free block to use** (or reject the request).

Everything else - splitting the block, marking it busy, merging free
neighbours - is done by :class:`~simulation.memory_manager.MemoryManager`.
That is why the four algorithms are so short: they genuinely are only a
*selection policy*.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # imported for type hints only - avoids a circular import
    from simulation.memory_manager import MemoryBlock


class AllocationStrategy(ABC):
    """Base class for all allocation algorithms.

    Subclasses must implement :meth:`select_block`. The framework calls
    :meth:`reset` before every simulation run so that stateful algorithms
    (Next Fit) start from a clean slate.
    """

    #: Human-readable name used in the UI and result tables.
    name: str = "Base"
    #: One-line description shown in the "about this algorithm" panel.
    description: str = ""
    #: Whether the algorithm keeps state between allocations.
    is_stateful: bool = False

    def __init__(self) -> None:
        self.last_examined_count: int = 0

    # -- interface -------------------------------------------------------
    @abstractmethod
    def select_block(
        self, blocks: List[MemoryBlock], size: int
    ) -> Optional[int]:
        """Return the index of the free block to use, or ``None`` if none fit.

        Implementations must set ``self.last_examined_count`` to the number
        of blocks they inspected, which feeds the "search operations" metric.
        """

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> None:
        """Clear any per-run state. Stateless algorithms ignore this."""
        self.last_examined_count = 0

    # -- helpers for subclasses -----------------------------------------
    @staticmethod
    def _fits(block: MemoryBlock, size: int) -> bool:
        return block.is_free and block.size >= size

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"
