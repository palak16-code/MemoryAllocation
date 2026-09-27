"""
analysis/__init__.py
-------------------
Public surface of the analysis package.
"""

from analysis.metrics import (
    METRIC_DEFINITIONS,
    average_search_cost,
    compute_metrics,
    external_fragmentation_index,
    to_comparison_table,
    utilisation_percent,
)
from analysis.visualization import (
    comparison_bar_figure,
    free_blocks_timeline_figure,
    layout_dataframe,
    memory_map_figure,
    utilisation_timeline_figure,
)

__all__ = [
    "METRIC_DEFINITIONS",
    "average_search_cost",
    "compute_metrics",
    "external_fragmentation_index",
    "to_comparison_table",
    "utilisation_percent",
    "comparison_bar_figure",
    "free_blocks_timeline_figure",
    "layout_dataframe",
    "memory_map_figure",
    "utilisation_timeline_figure",
]
