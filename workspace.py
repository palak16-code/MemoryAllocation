"""
workspace.py
------------
Owns the workspace tab bar, the editable process workload and the stored
result of every simulation run.

All state lives in one plain dictionary under a single session-state key:

    tabs              ordered list of open run/analysis tab descriptors
    active            id of the tab shown, or HOME_TAB for the input page
    workload          the process table the user is editing (a plain DataFrame)
    runs              run id      -> result dictionary
    analyses          analysis id -> comparison dictionary
    run_counter       number of runs created, for the "Run 1" labels
    analysis_counter  number of comparison tabs created

Everything here is ordinary Python data. The process table is deliberately
kept *outside* Streamlit's widget state: the ``st.data_editor`` on the Home tab
is only an editor for it, and every run receives its own deep copy, so closing
tabs, switching tabs or running again can never lose a row.

The allocation itself is still done by :func:`simulation.simulator.run_simulation`
and the unchanged algorithms in ``algorithms/``. This module converts their
output into plain data and keeps it around.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from algorithms import ALGORITHM_ORDER
from simulation import Process, run_simulation

#: Session-state key holding the whole workspace dictionary.
WORKSPACE_KEY = "workspace"

#: Id of the tab that always stays open and is never closed.
HOME_TAB = "home"

#: Separator used in tab labels, e.g. "Run 1 - Best Fit".
DASH = "\u2013"

#: Columns of the algorithm comparison table, in display order.
COMPARISON_COLUMNS = [
    "Algorithm",
    "Successful Allocations",
    "Failed Allocations",
    "Memory Utilization",
    "Free Memory",
]

#: Column names of the editable process table.
ID_COLUMN = "Process ID"
SIZE_COLUMN = "Memory Required (KB)"
ARRIVAL_COLUMN = "Arrival Time"
DURATION_COLUMN = "Duration"

DEFAULT_WORKLOAD = pd.DataFrame(
    [
        {ID_COLUMN: "P1", SIZE_COLUMN: 200, ARRIVAL_COLUMN: 0, DURATION_COLUMN: 5},
        {ID_COLUMN: "P2", SIZE_COLUMN: 150, ARRIVAL_COLUMN: 1, DURATION_COLUMN: 3},
        {ID_COLUMN: "P3", SIZE_COLUMN: 300, ARRIVAL_COLUMN: 2, DURATION_COLUMN: 4},
        {ID_COLUMN: "P4", SIZE_COLUMN: 100, ARRIVAL_COLUMN: 3, DURATION_COLUMN: 2},
    ],
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def get_state() -> dict:
    """Return the workspace dictionary, creating it on first use."""
    if WORKSPACE_KEY not in st.session_state:
        st.session_state[WORKSPACE_KEY] = {
            "tabs": [],
            "active": HOME_TAB,
            "workload": DEFAULT_WORKLOAD.copy(deep=True),
            "runs": {},
            "analyses": {},
            "run_counter": 0,
            "analysis_counter": 0,
        }
    return st.session_state[WORKSPACE_KEY]


def active_tab() -> Optional[dict]:
    """The open tab whose content should be shown, or ``None`` for the Home page.

    Home is not a tab in the bar - it is reached through the small house button
    - so "no tab selected" simply means the input page is being shown.
    """
    state = get_state()
    if state["active"] == HOME_TAB:
        return None
    for tab in state["tabs"]:
        if tab["id"] == state["active"]:
            return tab
    return None


def show_tab(tab_id: str) -> None:
    """Bring one tab to the front, or return to Home with ``HOME_TAB``."""
    get_state()["active"] = tab_id


def close_tab(tab_id: str) -> None:
    """Remove a tab and its stored result."""
    state = get_state()
    for index, tab in enumerate(state["tabs"]):
        if tab["id"] != tab_id:
            continue
        del state["tabs"][index]
        state["runs"].pop(tab_id, None)
        state["analyses"].pop(tab_id, None)
        if state["active"] == tab_id:
            # Prefer the tab on the left; fall back to the Home page.
            state["active"] = (
                state["tabs"][index - 1]["id"] if index > 0 else HOME_TAB
            )
        return


# ---------------------------------------------------------------------------
# The editable workload
# ---------------------------------------------------------------------------
def get_workload() -> pd.DataFrame:
    """The process table currently being edited (plain data, not widget state)."""
    return get_state()["workload"]


def set_workload(table: pd.DataFrame) -> None:
    """Store a copy of what the process table widget currently shows."""
    get_state()["workload"] = copy.deepcopy(table)


def snapshot_workload() -> List[dict]:
    """An independent plain-data copy of the workload, for one simulation.

    Every run keeps its own snapshot, so later edits on the Home tab can never
    change the workload a stored run was produced from.
    """
    return [
        {
            "pid": str(row[ID_COLUMN]),
            "size": int(row[SIZE_COLUMN]),
            "arrival": int(row[ARRIVAL_COLUMN]),
            "duration": int(row[DURATION_COLUMN]),
        }
        for row in get_workload().to_dict("records")
    ]


# ---------------------------------------------------------------------------
# Running simulations
# ---------------------------------------------------------------------------
def _representative_snapshot(result) -> object:
    """The snapshot that best shows where the algorithm placed the processes.

    The snapshot holding the largest number of running processes is preferred,
    because that is the moment where the most placements are visible at once;
    ties are broken by more occupied memory and then by the earliest time. Using
    the busiest moment matters - after the last process has finished, memory is
    empty again and the map would prove nothing.
    """
    return max(
        result.history,
        key=lambda s: (len(s.running), s.used_memory, -s.time),
    )


def _failures_at(result, total_size: int) -> List[dict]:
    """Explain every failed allocation using the state *at the time it failed*.

    The memory state of the whole run cannot answer "why was this process
    rejected": by the end of the run other processes may have finished and
    freed memory that did not exist when the request was made. Each failed
    attempt is therefore matched to the snapshot of its own time unit, which is
    the only state that is actually correct for that moment.
    """
    failures = []
    for record in result.records:
        if record.get("action") != "Allocate" or record.get("success"):
            continue
        snapshot = next(
            (s for s in result.history if s.time == record["time"]), None
        )
        if snapshot is None:
            continue
        failures.append(
            {
                "pid": record["pid"],
                "size": record["size"],
                "arrival": record["arrival"],
                "time": record["time"],
                "reason": record["reason"],
                "largest_free_block": snapshot.largest_free_block,
                "total_free": snapshot.free_memory,
                "total_memory": total_size,
                "blocks": [
                    {
                        "label": b.label(),
                        "start": b.start,
                        "end": b.end,
                        "size": b.size,
                        "status": "Free" if b.is_free else "Allocated",
                    }
                    for b in snapshot.blocks
                ],
                "holes": [
                    {"start": b.start, "end": b.end, "size": b.size}
                    for b in snapshot.blocks
                    if b.is_free
                ],
            }
        )
    return failures


def run_workload(
    total_size: int, processes: Sequence[Process], algorithm: str
) -> dict:
    """Run one workload with one algorithm and return a plain result dict."""
    result = run_simulation(
        total_size=total_size, processes=processes, algorithm=algorithm
    )
    shown = _representative_snapshot(result)

    return {
        "algorithm": algorithm,
        "total_memory": total_size,
        "processes": [p.to_dict() for p in processes],
        "blocks": [
            {
                "label": block.label(),
                "start": block.start,
                "end": block.end,
                "size": block.size,
                "status": "Free" if block.is_free else "Allocated",
            }
            for block in shown.blocks
        ],
        "shown_time": shown.time,
        "allocated": shown.used_memory,
        "free": shown.free_memory,
        "utilisation": shown.utilisation,
        "free_blocks": sum(1 for b in shown.blocks if b.is_free),
        "largest_free_block": shown.largest_free_block,
        "successes": result.metrics["processes_allocated"],
        "failures_count": result.metrics["processes_rejected"],
        "failures": _failures_at(result, total_size),
    }


def add_run(
    total_size: int, processes: Sequence[Process], algorithm: str
) -> str:
    """Run a workload, open a new Run tab for it and show that tab."""
    state = get_state()
    state["run_counter"] += 1
    run_id = f"run{state['run_counter']}"

    state["runs"][run_id] = run_workload(total_size, processes, algorithm)
    state["tabs"].append(
        {
            "id": run_id,
            "label": f"Run {state['run_counter']} {DASH} {algorithm}",
            "closable": True,
            "kind": "run",
        }
    )
    state["active"] = run_id
    return run_id


def add_analysis(total_size: int, processes: Sequence[Process]) -> str:
    """Run one workload with all four algorithms and open an Analysis tab."""
    state = get_state()
    state["analysis_counter"] += 1
    number = state["analysis_counter"]
    analysis_id = f"analysis{number}"

    rows: List[dict] = []
    for algorithm in ALGORITHM_ORDER:
        result = run_workload(total_size, processes, algorithm)
        rows.append(
            {
                "Algorithm": result["algorithm"],
                "Successful Allocations": result["successes"],
                "Failed Allocations": result["failures_count"],
                "Memory Utilization": result["utilisation"],
                "Free Memory": result["free"],
                "free_blocks": result["free_blocks"],
                "largest_free_block": result["largest_free_block"],
                "allocated": result["allocated"],
                "shown_time": result["shown_time"],
                "reject_note": reject_note(result["failures"]),
            }
        )

    state["analyses"][analysis_id] = {
        "total_memory": total_size,
        "process_count": len(processes),
        "rows": rows,
    }
    state["tabs"].append(
        {
            "id": analysis_id,
            "label": "Analysis" if number == 1 else f"Analysis {number}",
            "closable": True,
            "kind": "analysis",
        }
    )
    state["active"] = analysis_id
    return analysis_id


def processes_of(run: Dict[str, dict]) -> List[Process]:
    """Rebuild ``Process`` objects from the workload stored in a run."""
    return [Process(**entry) for entry in run["processes"]]


def _names_at(rows: List[dict], field: str, value) -> List[str]:
    """Names of every algorithm holding ``value`` in ``field``."""
    return [row["Algorithm"] for row in rows if row[field] == value]


def _joined(names: Sequence[str]) -> str:
    """\"A\", \"A and B\", \"A, B and C\"."""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def reject_note(failures: Sequence[dict]) -> str:
    """One short line saying what went wrong, taken from the numbers.

    Kept deliberately short: the overlay shows the evidence as cards, so the
    text only has to name the outcome.
    """
    if not failures:
        return "Every process in the workload was allocated."
    first = failures[0]
    extra = f" (+{len(failures) - 1} more)" if len(failures) > 1 else ""
    return (
        f"{first['pid']} was rejected: {first['size']} KB requested, but the "
        f"largest contiguous free block was {first['largest_free_block']} KB"
        f"{extra}."
    )


def summary_lines(analysis: dict) -> List[str]:
    """The comparison as one aligned line per algorithm - figures, not prose."""
    rows = analysis["rows"]
    width = max(len(row["Algorithm"]) for row in rows)

    lines = [
        f"WORKLOAD   {analysis['process_count']} processes  |  "
        f"{analysis['total_memory']} KB memory",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['Algorithm']:<{width}}   "
            f"{row['Successful Allocations']} successful  |  "
            f"{row['Failed Allocations']} failed  |  "
            f"{row['Memory Utilization']}% used"
        )
    return lines


def verdict(analysis: dict) -> str:
    """A single factual sentence about how the four algorithms compared.

    No algorithm is called better unless these numbers support it, and equal
    results are reported as equal.
    """
    rows = analysis["rows"]
    same = {
        (row["Successful Allocations"], row["Failed Allocations"],
         row["Memory Utilization"], row["Free Memory"])
        for row in rows
    }
    if len(same) == 1:
        return "All four algorithms produced the same allocation result for this workload."

    most, fewest = (
        max(rows, key=lambda r: r["Successful Allocations"]),
        min(rows, key=lambda r: r["Successful Allocations"]),
    )
    if most["Successful Allocations"] == fewest["Successful Allocations"]:
        return (
            "Every algorithm allocated the same number of processes, so the "
            "choice of algorithm did not change how many were admitted."
        )

    leaders = _joined(_names_at(rows, "Successful Allocations",
                                most["Successful Allocations"]))
    laggards = _joined(_names_at(rows, "Successful Allocations",
                                 fewest["Successful Allocations"]))
    return (
        f"{leaders} admitted the most processes "
        f"({most['Successful Allocations']}); {laggards} admitted the fewest "
        f"({fewest['Successful Allocations']})."
    )


# ---------------------------------------------------------------------------
# Tab bar
# ---------------------------------------------------------------------------
#: ``st.columns`` widths: a label needs most of the space, a close button only
#: needs a small square.
_LABEL_WIDTH = 4
_CLOSE_WIDTH = 1
#: Width of the round Home button.
_HOME_WIDTH = 1

#: Key of the container the tab bar lives in, which also scopes the CSS below.
_TAB_BAR_KEY = "tab_bar"

#: One small block of CSS, scoped to the tab bar container. It only lines the
#: buttons up with a hairline and marks the active one; nothing else is styled.
_CSS = """
<style>
.st-key-tab_bar div[data-testid="stButton"] button[kind="primary"] {
    border-bottom-right-radius: 0px;
    border-bottom-left-radius: 0px;
    border-bottom: 2px solid var(--primary-color, #ff4b4b);
}
.st-key-tab_bar div[data-testid="stHorizontalBlock"] {
    border-bottom: 1px solid rgba(128, 128, 128, 0.35);
}
</style>
"""


def render_tab_bar() -> None:
    """Draw the tab bar and apply whichever tab or close button was pressed.

    ``st.tabs`` cannot close an individual tab, so the bar is built from plain
    buttons. The Home page is not part of the bar - it is reached with the small
    house button on the left - so ``tabs`` holds only run and analysis tabs. One
    ``st.rerun`` is issued after a click so the active tab is highlighted on the
    very same run that shows its content.
    """
    state = get_state()

    st.html(_CSS)
    widths: List[int] = [_HOME_WIDTH]
    for tab in state["tabs"]:
        widths.append(_LABEL_WIDTH)
        if tab["closable"]:
            widths.append(_CLOSE_WIDTH)

    with st.container(key=_TAB_BAR_KEY):
        columns = st.columns(widths)

        clicked = False
        if columns[0].button(
            "\u2302",  # a small house outline
            key="home",
            help="Go to the Home page",
            type="primary" if state["active"] == HOME_TAB else "secondary",
        ):
            state["active"] = HOME_TAB
            clicked = True

        position = 1
        for tab in state["tabs"]:
            if columns[position].button(
                tab["label"],
                key=f"tab::{tab['id']}",
                type="primary" if state["active"] == tab["id"] else "secondary",
                width="stretch",
            ):
                state["active"] = tab["id"]
                clicked = True
            position += 1

            if tab["closable"]:
                if columns[position].button(
                    "\u00d7",
                    key=f"close::{tab['id']}",
                    help=f"Close {tab['label']}",
                    width="content",
                ):
                    close_tab(tab["id"])
                    clicked = True
                position += 1

    if clicked:
        st.rerun()
