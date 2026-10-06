"""
cards.py
--------
The single visual element the simulator needs: a proportional memory map.

The bar is drawn from the same block rows the tables are built from, so the
picture and the numbers can never disagree. Nothing is computed here.

Only two fills are used - allocated and free - and one amber inset marks the
largest contiguous free block, because that single block decides whether the
next request can be satisfied. Exact addresses are left to the tables.
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
.ma-map {
    display: flex;
    width: 100%;
    height: 34px;
    border: 1px solid rgba(128, 128, 128, 0.45);
    overflow: hidden;
}
.ma-seg {
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 0;
    overflow: hidden;
    white-space: nowrap;
    font-size: 0.68rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    border-right: 1px solid rgba(128, 128, 128, 0.30);
}
.ma-used { background: rgba(59, 130, 246, 0.28); }
.ma-free { background: rgba(148, 163, 184, 0.22); }
.ma-largest { box-shadow: inset 0 0 0 2px rgba(245, 158, 11, 0.90); }
.ma-key {
    font-size: 0.72rem;
    opacity: 0.70;
    margin-top: 5px;
}
.ma-chip {
    display: inline-block;
    width: 9px;
    height: 9px;
    margin-right: 5px;
}
</style>
"""


def render_styles() -> None:
    """Inject the style block. Called once, at the top of the page."""
    st.html(STYLES)


def memory_map(blocks: Sequence[dict], largest_free: Optional[int] = None) -> None:
    """Draw memory left to right, each block in proportion to its size.

    ``blocks`` are the rows of a memory-state table. When ``largest_free`` is
    given, the free block of that size is outlined - it is the block an
    allocator would have to use for the largest possible request.
    """
    total = sum(block["size"] for block in blocks) or 1

    segments = ['<div class="ma-map">']
    for block in blocks:
        is_free = block["status"] == "Free"
        css = "ma-seg ma-free" if is_free else "ma-seg ma-used"
        if is_free and block["size"] == largest_free:
            css += " ma-largest"
        # "FREE 150" reads badly and the key below already says what a free
        # segment is, so free segments carry only their size.
        if block["size"] / total < _LABEL_SHARE:
            label = ""
        elif is_free:
            label = str(block["size"])
        else:
            label = f"{block['label']} {block['size']}"
        owner = "free block" if is_free else str(block["label"])
        segments.append(
            f'<div class="{css}" style="flex:{block["size"]} 1 0;" '
            f'title="{escape(owner)}: {block["start"]}-{block["end"]} KB, '
            f'{block["size"]} KB">{escape(label)}</div>'
        )
    segments.append("</div>")

    key = [
        '<div class="ma-key">',
        '<span><span class="ma-chip ma-used"></span>allocated</span>',
        '<span><span class="ma-chip ma-free"></span>free</span>',
    ]
    if largest_free:
        key.append(
            f'<span><span class="ma-chip ma-largest"></span>'
            f"largest contiguous free block: {largest_free} KB</span>"
        )
    key.append("</div>")

    st.html("".join(segments))
    st.html("".join(key))