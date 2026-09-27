"""
visualization.py
----------------
Plotly figures for the memory map and the algorithm comparison.

The memory map is built from the *same* ``MemoryBlock`` objects that the
simulator uses, so the picture can never disagree with the underlying
state - there is no second, hand-maintained copy of the layout.

Figure technique
----------------
Each block is drawn as a segment of a single horizontal stacked bar: the
segment's width is its size in KB and its offset is ``block.start``. Using
``base=`` lets a segment start at a non-zero address, which is what makes a
proportional address-space map possible.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import plotly.graph_objects as go

from simulation.memory_manager import MemoryBlock

# --- colours ---------------------------------------------------------------
FREE_COLOR = "#B0BEC5"          # blue-grey for free memory
FREE_EDGE = "#78909C"
HIGHLIGHT_EDGE = "#D32F2F"      # red outline for the selected block
PALETTE = [
    "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
    "#76B7B2", "#EDC948", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#86BCB6", "#D37295", "#A0CBE8", "#FFBE7D", "#8CD17D",
]


def _color_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def block_figure_text(block: MemoryBlock) -> str:
    """Label drawn inside a block: process ID (or FREE) and its size."""
    return f"{block.label()}<br>{block.size} KB"


def memory_map_figure(
    blocks: Sequence[MemoryBlock],
    total_size: Optional[int] = None,
    title: str = "Memory Layout",
    highlight_pid: Optional[str] = None,
    height: int = 190,
) -> go.Figure:
    """Draw the memory map for one point in time.

    Parameters
    ----------
    blocks:
        The current block list (from ``MemoryManager.snapshot()``).
    total_size:
        Width of the address space; defaults to the end of the last block.
    highlight_pid:
        When given, that process' block gets a thick red outline, used to
        show "the block chosen by this allocation".
    """
    fig = go.Figure()
    process_index: Dict[str, int] = {}

    for block in blocks:
        if block.is_free:
            color, edge, line_width = FREE_COLOR, FREE_EDGE, 1
            label = block_figure_text(block)
        else:
            if block.pid not in process_index:
                process_index[block.pid] = len(process_index)
            color = _color_for(process_index[block.pid])
            edge, line_width = "#37474F", 1
            label = f"<b>{block.pid}</b><br>{block.size} KB"
            if highlight_pid and block.pid == highlight_pid:
                edge, line_width = HIGHLIGHT_EDGE, 3

        fig.add_trace(
            go.Bar(
                x=[block.size],
                y=["Memory"],
                base=[block.start],
                orientation="h",
                marker=dict(
                    color=color,
                    line=dict(color=edge, width=line_width),
                ),
                text=[label],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="#FFFFFF" if not block.is_free else "#37474F",
                              size=12),
                hovertemplate=(
                    f"<b>{block.label()}</b><br>"
                    f"Start: {block.start} KB<br>"
                    f"End: {block.end} KB<br>"
                    f"Size: {block.size} KB"
                    "<extra></extra>"
                ),
                showlegend=False,
                name=block.label(),
            )
        )

    span = total_size or (blocks[-1].end if blocks else 1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        barmode="stack",
        height=height,
        margin=dict(l=10, r=10, t=45, b=40),
        xaxis=dict(
            title="Address space (KB)",
            range=[0, span],
            showgrid=True,
            gridcolor="#ECEFF1",
        ),
        yaxis=dict(showticklabels=False, range=[-0.6, 0.6]),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def legend_figure(blocks: Sequence[MemoryBlock]) -> go.Figure:
    """Small colour legend matching the memory-map colours."""
    seen: Dict[str, str] = {}
    for block in blocks:
        if block.is_free:
            seen.setdefault("Free memory", FREE_COLOR)
        else:
            idx = len([k for k in seen if k != "Free memory"])
            seen.setdefault(block.pid, _color_for(idx))

    fig = go.Figure()
    for label, color in seen.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=label, hoverinfo="skip",
            )
        )
    fig.update_layout(
        height=60, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.4,
                    xanchor="left", x=0, font=dict(size=11)),
        showlegend=True,
    )
    return fig


def layout_dataframe(blocks: Sequence[MemoryBlock]):
    """Tabular view of the layout (built on the shared block list)."""
    import pandas as pd

    rows = []
    for order, block in enumerate(blocks):
        rows.append(
            {
                "#": order,
                "Start (KB)": block.start,
                "End (KB)": block.end,
                "Size (KB)": block.size,
                "Owner": block.label(),
                "Type": "Free" if block.is_free else "Process",
                "Start %": round(block.start / block.end * 100, 1)
                if block.end else 0.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison charts
# ---------------------------------------------------------------------------

def comparison_bar_figure(
    results: Dict[str, "object"],
    metric: str,
    pretty_name: str,
    higher_is_better: Optional[bool] = None,
) -> go.Figure:
    """Bar chart of one metric across algorithms."""
    from analysis.metrics import compute_metrics  # noqa: F401  (docs link)

    names = list(results.keys())
    values = [results[name].metrics.get(metric, 0) for name in names]

    colors = []
    best = None
    if higher_is_better is not None and values:
        best = max(values) if higher_is_better else min(values)
    for value in values:
        if best is not None and value == best:
            colors.append("#59A14F")     # green = best in this metric
        else:
            colors.append("#4E79A7")

    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            text=[str(v) for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=pretty_name,
        xaxis_title="Allocation algorithm",
        yaxis_title=pretty_name,
        showlegend=False,
        height=330,
        margin=dict(l=10, r=10, t=45, b=40),
        plot_bgcolor="white",
    )
    return fig


def utilisation_timeline_figure(
    results: Dict[str, "object"], total_size: int
) -> go.Figure:
    """Line chart of memory utilisation over time, one line per algorithm."""
    fig = go.Figure()
    for name, result in results.items():
        times, utils = [], []
        for snap in result.history:
            times.append(snap.time)
            utils.append(
                round(snap.used_memory / total_size * 100, 2) if total_size else 0
            )
        fig.add_trace(
            go.Scatter(
                x=times, y=utils, mode="lines+markers", name=name,
                line=dict(width=2), marker=dict(size=6),
            )
        )
    fig.update_layout(
        title="Memory utilisation over time",
        xaxis_title="Time (units)",
        yaxis_title="Utilisation (%)",
        hovermode="x unified",
        height=360,
        margin=dict(l=10, r=10, t=45, b=40),
        plot_bgcolor="white",
    )
    return fig


def free_blocks_timeline_figure(
    results: Dict[str, "object"],
) -> go.Figure:
    """Line chart of the number of free blocks over time."""
    fig = go.Figure()
    for name, result in results.items():
        fig.add_trace(
            go.Scatter(
                x=[s.time for s in result.history],
                y=[s.free_blocks for s in result.history],
                mode="lines+markers", name=name,
                line=dict(width=2), marker=dict(size=6),
            )
        )
    fig.update_layout(
        title="Number of free blocks over time",
        xaxis_title="Time (units)",
        yaxis_title="Free blocks",
        hovermode="x unified",
        height=360,
        margin=dict(l=10, r=10, t=45, b=40),
        plot_bgcolor="white",
    )
    return fig
