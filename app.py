"""
app.py
------
Streamlit interface for the Memory Allocation Simulator.

Three pages, all reached from one tab bar:

* **Home** collects the memory size, the process table and the algorithm, and
  opens a new Run tab when *Run Simulation* is pressed. A small house button in
  the tab bar always leads back here; Home is not itself a tab.
* **Run n - <algorithm>** shows the memory map and table at the busiest moment
  of the run, a row of figures, and a *Detailed Metrics* button. The detail
  opens as an overlay so the run page underneath is never crowded.
* **Analysis** replays one shared workload through all four algorithms and
  tabulates the outcome.

The process table is stored in ``workspace.py`` as plain data, never as widget
state, so every run owns an independent snapshot of the workload it was
produced from. Nothing in this file allocates memory: the algorithms live in
``algorithms/`` and the event loop in ``simulation/simulator.py``.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import cards
import workspace
from algorithms import ALGORITHM_ORDER
from simulation import Process, WorkloadError, validate_workload
from workspace import (
    ARRIVAL_COLUMN,
    COMPARISON_COLUMNS,
    DURATION_COLUMN,
    ID_COLUMN,
    SIZE_COLUMN,
)

st.set_page_config(page_title="Memory Allocation Simulator", layout="wide")

#: Default physical memory in KB.
DEFAULT_MEMORY = 1000

#: Session-state key holding the id of the run whose overlay is open.
DETAIL_KEY = "detailed_run"

#: Column order of the memory-state table on a Run tab.
MEMORY_COLUMNS = {
    "label": "Block / Process",
    "start": "Start Address",
    "end": "End Address",
    "size": "Size (KB)",
    "status": "Status",
}

#: Column order of the free-block table inside the overlay.
HOLE_COLUMNS = {
    "block": "Free Block",
    "start": "Start",
    "end": "End",
    "size": "Size (KB)",
}

PROCESS_COLUMN_CONFIG = {
    ID_COLUMN: st.column_config.TextColumn(ID_COLUMN, required=True),
    SIZE_COLUMN: st.column_config.NumberColumn(
        SIZE_COLUMN, min_value=0, step=10, required=True
    ),
    ARRIVAL_COLUMN: st.column_config.NumberColumn(
        ARRIVAL_COLUMN, min_value=0, step=1, required=True
    ),
    DURATION_COLUMN: st.column_config.NumberColumn(
        DURATION_COLUMN, min_value=0, step=1, required=True
    ),
}

#: The one idea a rejected allocation has to convey.
EXTERNAL_FRAGMENTATION = (
    "External fragmentation: the free memory exists, but it is split into "
    "separate blocks. Dynamic partitioning needs a single contiguous block, "
    "so the request cannot be served."
)
OUT_OF_MEMORY = (
    "The system was genuinely out of memory at that moment - the free memory "
    "was smaller than the request."
)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def read_processes(table: pd.DataFrame) -> tuple[list[Process], list[str]]:
    """Convert the process table into ``Process`` objects.

    Returns ``(processes, problems)``. A non-empty ``problems`` list means the
    input is invalid and no simulation should be run.
    """
    processes: list[Process] = []
    problems: list[str] = []

    for position, row in enumerate(table.to_dict("records"), start=1):
        raw_id = row[ID_COLUMN]
        pid = "" if pd.isna(raw_id) else str(raw_id).strip()
        size = row[SIZE_COLUMN]
        arrival = row[ARRIVAL_COLUMN]
        duration = row[DURATION_COLUMN]

        if not pid or any(pd.isna(value) for value in (size, arrival, duration)):
            problems.append(
                f"Row {position}: fill in {ID_COLUMN}, {SIZE_COLUMN}, "
                f"{ARRIVAL_COLUMN} and {DURATION_COLUMN}."
            )
            continue

        try:
            processes.append(
                Process(
                    pid,
                    int(size),
                    arrival=int(arrival),
                    duration=int(duration),
                )
            )
        except WorkloadError as error:
            problems.append(str(error))

    problems.extend(validate_workload(processes))
    return processes, problems


# ---------------------------------------------------------------------------
# Detailed metrics overlay
# ---------------------------------------------------------------------------
def show_failure(failure: dict) -> None:
    """One rejected process: the numbers, the layout, and the one reason.

    Everything shown is the state of memory at the moment of the attempt, so
    the figures cannot be explained away by later activity in the run.
    """
    st.subheader(f"Allocation Failed: {failure['pid']}")

    st.table(
        pd.DataFrame(
            [
                {
                    "Process ID": failure["pid"],
                    "Memory Required (KB)": failure["size"],
                    "Arrival Time": failure["arrival"],
                    "Attempt Time": failure["time"],
                    "Total Free Memory (KB)": failure["total_free"],
                    "Largest Contiguous Free Block (KB)":
                        failure["largest_free_block"],
                }
            ]
        )
    )

    st.markdown("**Memory at the moment of the attempt**")
    cards.memory_map(failure["blocks"], failure["largest_free_block"])

    if failure["holes"]:
        st.table(
            pd.DataFrame(
                [
                    {
                        "block": f"H{position}",
                        "start": hole["start"],
                        "end": hole["end"],
                        "size": hole["size"],
                    }
                    for position, hole in enumerate(failure["holes"], start=1)
                ]
            ).rename(columns=HOLE_COLUMNS)
        )
    else:
        st.write("No memory was free at that time.")

    st.code(
        f"Requested: {failure['size']} KB"
        f"   >   Largest contiguous block: {failure['largest_free_block']} KB",
        language=None,
    )
    if failure["total_free"] >= failure["size"]:
        st.caption(EXTERNAL_FRAGMENTATION)
    else:
        st.caption(OUT_OF_MEMORY)


@st.dialog("Detailed Metrics", width="large", dismissible=False)
def show_detailed_metrics(run: dict) -> None:
    """The full-screen view of one run.

    Each rejected process is described once, with only the figures needed to
    account for the rejection. The dialog is not dismissible so that it can
    only be left through the Close button, which clears the session flag and
    reruns the app - the Run tab underneath is then exactly as the user left it.
    """
    title, close = st.columns([6, 1])
    title.caption(
        f"{run['algorithm']}  |  {len(run['processes'])} processes  |  "
        f"{run['total_memory']} KB memory"
    )
    if close.button("Close \u2715", key="detail_close_top", width="stretch"):
        close_details()

    if run["failures"]:
        for position, failure in enumerate(run["failures"]):
            if position:
                st.divider()
            show_failure(failure)
    else:
        st.write(
            f"All {len(run['processes'])} processes were allocated - there is "
            "nothing to explain."
        )
        show_metrics(
            {
                "Processes allocated": f"{run['successes']}/{len(run['processes'])}",
                "Memory utilization": f"{run['utilisation']}%",
                "Free memory": f"{run['free']} KB",
                "Contiguous free blocks": run["free_blocks"],
                "Largest free block": f"{run['largest_free_block']} KB",
            }
        )

    st.divider()
    if st.button("Close", key="detail_close_bottom"):
        close_details()


def close_details() -> None:
    """Leave the overlay and return to the run page underneath."""
    st.session_state[DETAIL_KEY] = None
    st.rerun()


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
def show_metrics(values: dict[str, str]) -> None:
    """One row of labelled metrics."""
    for column, (label, value) in zip(st.columns(len(values)), values.items()):
        column.metric(label, value)


def render_home() -> None:
    """The Home page: memory configuration, process input, algorithm, run."""
    st.subheader("Memory Configuration")
    total_memory = st.number_input(
        "Total memory (KB)", min_value=1, value=DEFAULT_MEMORY, step=50
    )

    st.subheader("Process Input")
    st.caption("Add or remove rows as needed. All four fields are required.")
    # The widget is only an editor: whatever it shows is copied straight into
    # the workspace, and a run takes its own snapshot of that copy.
    workspace.set_workload(
        st.data_editor(
            workspace.get_workload(),
            key="home_process_table",
            num_rows="dynamic",
            hide_index=True,
            column_config=PROCESS_COLUMN_CONFIG,
        )
    )

    st.subheader("Algorithm Selection")
    algorithm = st.radio("Allocation algorithm", ALGORITHM_ORDER)

    if st.button("Run Simulation", type="primary"):
        processes, problems = read_processes(workspace.get_workload())
        if problems:
            st.error("Please fix the following before running the simulation:")
            for problem in problems:
                st.error(f"- {problem}")
        else:
            workspace.add_run(int(total_memory), processes, algorithm)
            st.rerun()


def render_run(tab: dict, run: dict) -> None:
    """A Run tab: memory map and table, figures, one button, the actions."""
    tab_id = tab["id"]

    st.subheader(f"Run: {run['algorithm']}")
    st.caption(
        f"Memory at its most crowded moment - {len(run['blocks'])} blocks, "
        f"reached at time unit {run['shown_time']}."
    )
    cards.memory_map(run["blocks"], run["largest_free_block"])
    st.table(pd.DataFrame(run["blocks"]).rename(columns=MEMORY_COLUMNS))

    show_metrics(
        {
            "Total memory": f"{run['total_memory']} KB",
            "Allocated": f"{run['allocated']} KB",
            "Free": f"{run['free']} KB",
            "Utilization": f"{run['utilisation']}%",
            "Allocated processes": run["successes"],
            "Rejected processes": run["failures_count"],
        }
    )

    if st.button("Detailed Metrics", key=f"detail::{tab_id}"):
        st.session_state[DETAIL_KEY] = tab_id
    if st.session_state.get(DETAIL_KEY) == tab_id:
        show_detailed_metrics(run)

    st.divider()
    left, right = st.columns(2)
    if left.button(
        "New workload", key=f"add::{tab_id}", width="stretch"
    ):
        workspace.show_tab(workspace.HOME_TAB)
        st.rerun()
    if right.button(
        "Compare all algorithms", key=f"compare::{tab_id}", width="stretch"
    ):
        workspace.add_analysis(run["total_memory"], workspace.processes_of(run))
        st.rerun()


def show_algorithm_cards(analysis: dict) -> None:
    """One bordered block per algorithm, showing the same facts as the table."""
    for row_start in range(0, len(analysis["rows"]), 2):
        pair = analysis["rows"][row_start : row_start + 2]
        for column, row in zip(st.columns(2), pair):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{row['Algorithm']}**")
                    show_metrics(
                        {
                            "Allocated": row["Successful Allocations"],
                            "Rejected": row["Failed Allocations"],
                            "Utilization": f"{row['Memory Utilization']}%",
                        }
                    )
                    st.caption(
                        f"Largest free block {row['largest_free_block']} KB in "
                        f"{row['free_blocks']} block(s)  |  "
                        f"{row['reject_note']}"
                    )


def render_analysis(analysis: dict) -> None:
    """An Analysis tab: the four algorithms compared on one shared workload."""
    st.subheader("Algorithm Comparison")
    st.table(pd.DataFrame(analysis["rows"])[COMPARISON_COLUMNS])

    st.caption(workspace.verdict(analysis))
    if not workspace.identical_results(analysis):
        # Four identical bars carry no information, so the chart is only drawn
        # when the algorithms actually disagreed on something.
        table = pd.DataFrame(analysis["rows"])[COMPARISON_COLUMNS]
        st.bar_chart(table, x="Algorithm", y="Memory Utilization",
                     color="Algorithm")

    show_algorithm_cards(analysis)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Memory Allocation Simulator")
st.caption(
    "Dynamic memory allocation in Operating Systems: "
    "First Fit - Best Fit - Worst Fit - Next Fit"
)

cards.render_styles()
workspace.render_tab_bar()

tab = workspace.active_tab()
if tab is None:
    render_home()
elif tab["kind"] == "run":
    render_run(tab, workspace.get_state()["runs"][tab["id"]])
else:
    render_analysis(workspace.get_state()["analyses"][tab["id"]])