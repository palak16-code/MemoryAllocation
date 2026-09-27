"""
algorithms/__init__.py
----------------------
Registry mapping the display name of an algorithm to its class.

The UI and the simulator never import concrete algorithm modules - they ask
for an instance by name, which keeps the rest of the code algorithm-agnostic
and makes it trivial to add a fifth algorithm later.
"""

from __future__ import annotations

from typing import Dict, List, Type

from algorithms.base import AllocationStrategy
from algorithms.best_fit import BestFit
from algorithms.first_fit import FirstFit
from algorithms.next_fit import NextFit
from algorithms.worst_fit import WorstFit

ALGORITHM_REGISTRY: Dict[str, Type[AllocationStrategy]] = {
    "First Fit": FirstFit,
    "Best Fit": BestFit,
    "Worst Fit": WorstFit,
    "Next Fit": NextFit,
}

#: Stable display order used by every table and chart.
ALGORITHM_ORDER: List[str] = list(ALGORITHM_REGISTRY.keys())


def available_algorithms() -> List[str]:
    """Names of all implemented algorithms, in presentation order."""
    return list(ALGORITHM_ORDER)


def get_algorithm(name: str) -> AllocationStrategy:
    """Create a *fresh* instance of the named algorithm.

    A new instance is returned on every call because Next Fit carries a
    roving pointer; reusing one instance across runs would leak state.
    """
    key = str(name).strip()
    if key not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. "
            f"Available: {', '.join(ALGORITHM_ORDER)}"
        )
    return ALGORITHM_REGISTRY[key]()


def algorithm_info() -> Dict[str, str]:
    """Display name -> description, for the algorithm reference panel."""
    return {name: cls.description for name, cls in ALGORITHM_REGISTRY.items()}


__all__ = [
    "AllocationStrategy",
    "FirstFit",
    "BestFit",
    "WorstFit",
    "NextFit",
    "ALGORITHM_REGISTRY",
    "ALGORITHM_ORDER",
    "available_algorithms",
    "get_algorithm",
    "algorithm_info",
]
