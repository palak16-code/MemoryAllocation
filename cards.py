"""
cards.py
--------
Small HTML renderers used to present simulation results visually.

This module is presentation only: it receives numbers and blocks that
``workspace.py`` already took from the simulator, and draws them. Nothing is
computed here.

:data:`STYLES` is the only custom CSS in the whole project. Every colour is a
translucent overlay rather than a fixed value, so the cards sit correctly on the
active Streamlit theme instead of fighting it. Colour only ever carries meaning:
amber marks the largest free block, red marks a rejected allocation.
"""

from __future__ import annotations

from html import escape
from typing import Optional, Sequence

import streamlit as st

#: Segments narrower than this share of memory get no text label, because the
#: label would not fit and would only be clipped.
_LABEL_SHARE = 0.09

STYLES = """
<style>
.ma-card {
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0 10px 0;
}
.ma-card-fail {
    border-color: rgba(239, 68, 68, 0.55);
    background: rgba(239, 68, 68, 0.12);
}
.ma-label {
    font-size: 0.70rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    opacity: 0.65;
    margin-bottom: 4px;
}
.ma-fail-text { color: rgba(248, 113, 113, 0.95); }
.ma-row {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    font-size: 0.85rem;
}
.ma-bar {
    display: flex;
    width: 100%;
    height: 40px;
    border: 1px solid rgba(128, 128, 128, 0.45);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 4px;
}
.ma-seg {
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 0;
    overflow: hidden;
    white-space: nowrap;
    font-size: 0.70rem;
    border-right: 1px solid rgba(128, 128, 128, 0.35);
}
.ma-used { background: rgba(59, 130, 246, 0.30); }
.ma-free { background: rgba(148, 163, 184, 0.26); }
.ma-biggest { background: rgba(245, 158, 11, 0.38); }
.ma-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    font-size: 0.75rem;
    opacity: 0.75;
    margin-bottom: 10px;
}
.ma-swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 5px;
}
.ma-compare {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    font-size: 1rem;
    padding: 6px 0;
}
.ma-op { font-size: 1.4rem; opacity: 0.85; }
.ma-verdict { color: rgba(248, 113, 113, 0.95); font-weight: 600; }
.ma-note {
    border-left: 3px solid rgba(245, 158, 11, 0.75);
    background: rgba(245, 158, 11, 0.10);
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    margin-bottom: 10px;
}
</style>
"""


def render_styles() -> None:
    """Inject the style block. Called once, at the top of the page."""
    st.html(STYLES)


def failure_banner(failure: dict) -> None:
    """The status card: who was rejected, what it wanted, when it tried."""
    st.html(
        f'<div class="ma-card ma-card-fail">'
        f'<div class="ma-label ma-fail-text">Allocation failed</div>'
        f'<div style="font-size:1.05rem;font-weight:600;">'
        f"{escape(str(failure['pid']))} &#10007; NOT ALLOCATED</div>"
        f'<div class="ma-row" style="margin-top:6px;">'
        f"<span>Requested <strong>{failure['size']} KB</strong></span>"
        f"<span>Arrival <strong>time {failure['arrival']}</strong></span>"
        f"<span>Attempted at <strong>time unit {failure['time']}</strong></span>"
        f"</div></div>"
    )


def memory_bar(blocks: Sequence[dict], largest_free: Optional[int] = None) -> None:
    """Memory at one instant, drawn left to right in proportion to size.

    ``blocks`` are the rows of a memory-state table. The free block whose size
    equals ``largest_free`` is highlighted in amber, because that single block
    is what decides whether the next request can be satisfied.
    """
    total = sum(block["size"] for block in blocks) or 1

    segments = ['<div class="ma-bar">']
    for block in blocks:
        is_free = block["status"] == "Free"
        css = "ma-seg ma-free" if is_free else "ma-seg ma-used"
        if is_free and block["size"] == largest_free:
            css += " ma-biggest"
        # "FREE 150" reads badly, and the colour legend already says what a
        # free block is, so only the size is shown on the free segments.
        if block["size"] / total < _LABEL_SHARE:
            label = ""
        elif is_free:
            label = f"{block['size']} free"
        else:
            label = f"{block['label']} {block['size']}"
        what = "Free block" if is_free else str(block["label"])
        segments.append(
            f'<div class="{css}" style="flex:{block["size"]} 1 0;" '
            f'title="{escape(what)} {block["start"]}-{block["end"]} KB, '
            f'{block["size"]} KB">{escape(label)}</div>'
        )
    segments.append("</div>")
    st.html("".join(segments))

    legend = [
        '<span><span class="ma-swatch ma-used"></span>allocated</span>',
        '<span><span class="ma-swatch ma-free"></span>free</span>',
    ]
    if largest_free:
        legend.insert(
            1,
            f'<span><span class="ma-swatch ma-biggest"></span>'
            f"largest contiguous free block: {largest_free} KB</span>",
        )
    st.html(f'<div class="ma-legend">{"".join(legend)}</div>')


def comparison(requested: int, largest_free: int) -> None:
    """The verdict, as a shape instead of a sentence."""
    st.html(
        f'<div class="ma-card">'
        f'<div class="ma-compare">'
        f"<span><strong>{requested} KB</strong> requested</span>"
        f'<span class="ma-op">&gt;</span>'
        f"<span><strong>{largest_free} KB</strong> largest contiguous block</span>"
        f"</div>"
        f'<div class="ma-verdict">'
        "&#10007; Allocation failed - no contiguous block was large enough."
        f"</div></div>"
    )


def note(text: str) -> None:
    """A short highlighted remark, for the one idea that needs emphasising."""
    st.html(f'<div class="ma-note">{escape(text)}</div>')


def options(items: Sequence[str]) -> None:
    """Short numbered options in a two-column grid of bordered cards."""
    for row_start in range(0, len(items), 2):
        row = list(enumerate(items[row_start : row_start + 2], start=row_start + 1))
        for column, (number, option) in zip(st.columns(2), row):
            with column:
                with st.container(border=True):
                    st.html(
                        f'<div class="ma-label">Option {number}</div>'
                        f'<div style="font-size:0.88rem;">{escape(option)}</div>'
                    )
