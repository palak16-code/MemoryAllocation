"""
app.py
------
Streamlit interface for the Memory Allocation Simulator, organised as a small
workspace of tabs:

* **Home** collects the memory size, the process table and the algorithm, and
  opens a new Run tab when *Run Simulation* is pressed. A small house button in
  the tab bar always leads back here; Home is not itself a tab.
* **Run n - <algorithm>** stays deliberately short: the memory-state table, the
  six basic metrics, and a *View Detailed Metrics* button. The detail lives in a
  full-screen overlay so the run page underneath is never crowded.
* **Analysis** compares the four algorithms on one shared workload, with a
  *View Analysis* button that turns the table into cards and a one-line verdict.

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
# Explaining a failed allocation
# ---------------------------------------------------------------------------
def failure_options(failure: dict, run: dict) -> list[str]:
    """Short, verified ways to make one rejected process fit.

    Each entry is one option card, so the wording stays short. Anything that is
    not checked against this run is stated conditionally, and no allocation
    policy is ever promised to succeed.
    """
    size = failure["size"]
    occupied = failure["total_memory"] - failure["total_free"]
    options = []

    needed = occupied + size
    if needed > failure["total_memory"]:
        options.append(
            f"Raise total memory to {needed} KB or more - {occupied} KB was "
            f"already in use at time {failure['time']}, so {needed} KB is the "
            f"smallest total that could hold everything at once."
        )
    else:
        options.append(
            f"Make the free space contiguous - {failure['total_free']} KB was "
            f"free at time {failure['time']}, but no single block reached "
            f"{size} KB."
        )

    options.append(
        f"Lower {failure['pid']}'s requirement to "
        f"{failure['largest_free_block']} KB or less, the largest block that "
        f"was free at that moment."
    )

    still_running = [
        entry
        for entry in run["processes"]
        if entry["pid"] != failure["pid"]
        and entry["arrival"] < failure["arrival"]
        and entry["arrival"] + entry["duration"] > failure["time"]
    ]
    if still_running:
        who = ", ".join(
            f"{entry['pid']} (finishes at time "
            f"{entry['arrival'] + entry['duration']})"
            for entry in still_running
        )
        options.append(
            f"Free memory sooner by shortening {who} - "
            f"{'it was' if len(still_running) == 1 else 'they were'} still "
            f"running when {failure['pid']} asked for space."
        )

    options.append(
        f"Compare policies with \"Run All Algorithms\" to see whether another "
        f"policy lays the blocks out differently. It may or may not admit "
        f"{failure['pid']}."
    )
    return options


def show_failure_cards(failure: dict, run: dict) -> None:
    """One rejected process, presented as evidence rather than as prose."""
    cards.failure_banner(failure)
    show_metrics(
        {
            "Total free memory": f"{failure['total_free']} KB",
            "Largest contiguous block": f"{failure['largest_free_block']} KB",
            "Process request": f"{failure['size']} KB",
        }
    )

    cards.memory_bar(failure["blocks"], failure["largest_free_block"])
    cards.comparison(failure["size"], failure["largest_free_block"])

    if failure["total_free"] >= failure["size"]:
        cards.note(
            f"Total free memory ({failure['total_free']} KB) was actually larger "
            f"than the {failure['size']} KB requested - it was split across "
            f"separate holes. Dynamic partitioning needs one contiguous block, so "
            f"this is external fragmentation."
        )
    else:
        cards.note(
            f"Memory was genuinely short at that moment: "
            f"{failure['total_free']} KB free against {failure['size']} KB "
            f"requested."
        )

    if failure["holes"]:
        st.markdown("**Contiguous free blocks at that time**")
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

    st.markdown("**How this process could be allocated**")
    cards.options(failure_options(failure, run))


@st.dialog("Detailed Metrics", width="large", dismissible=False)
def show_detailed_metrics(run: dict) -> None:
    """The full-screen view of one run: run-wide figures, then every failure.

    The dialog is not dismissible so that it can only be left through the Close
    button, which clears the session flag and reruns the app - the Run tab
    underneath is then exactly as the user left it.
    """
    left, right = st.columns([5, 1])
    left.caption(
        f"Run: {run['algorithm']}  |  {len(run['processes'])} processes  |  "
        f"{run['total_memory']} KB memory"
    )
    if right.button("Close \u2715", key="detail_close_top", width="stretch"):
        close_details()

    show_metrics(
        {
            "Contiguous free blocks": run["free_blocks"],
            "Largest free block": f"{run['largest_free_block']} KB",
            "Memory in use at peak": f"{run['allocated']} KB",
            "Peak reached at": f"time unit {run['shown_time']}",
        }
    )

    if run["failures"]:
        st.divider()
        st.caption(
            "Every figure below describes memory as it was when the attempt was "
            "made, not as it was at the end of the run."
        )
        for failure in run["failures"]:
            show_failure_cards(failure, run)
    else:
        st.success(
            "Every process in this workload found a contiguous block large enough, "
            "so there is nothing to explain."
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
    """A Run tab: memory-state table, six metrics, one button, the actions."""
    tab_id = tab["id"]

    st.subheader(f"Run: {run['algorithm']}")
    st.caption(
        f"Workload: {len(run['processes'])} processes in {run['total_memory']} KB "
        f"of memory."
    )

    st.subheader("Memory State")
    st.caption(
        f"Memory at its most crowded moment - {len(run['blocks'])} blocks, "
        f"reached at time unit {run['shown_time']}."
    )
    st.table(pd.DataFrame(run["blocks"]).rename(columns=MEMORY_COLUMNS))

    st.subheader("Metrics")
    show_metrics(
        {
            "Total memory": f"{run['total_memory']} KB",
            "Allocated memory": f"{run['allocated']} KB",
            "Free memory": f"{run['free']} KB",
            "Memory utilization": f"{run['utilisation']}%",
            "Successful allocations": run["successes"],
            "Failed allocations": run["failures_count"],
        }
    )

    if st.button(
        "View Detailed Metrics", key=f"detail::{tab_id}", type="primary"
    ):
        st.session_state[DETAIL_KEY] = tab_id
    if st.session_state.get(DETAIL_KEY) == tab_id:
        show_detailed_metrics(run)

    st.divider()
    st.subheader("Actions")
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("**Add more processes**")
            st.caption("Add or edit processes, then run the simulation again.")
            if st.button("Add More Processes", key=f"add::{tab_id}", width="stretch"):
                workspace.show_tab(workspace.HOME_TAB)
                st.rerun()

    with right:
        with st.container(border=True):
            st.markdown("**Compare all algorithms**")
            st.caption("Run the same workload through all four policies.")
            if st.button(
                "Run All Algorithms", key=f"compare::{tab_id}", width="stretch"
            ):
                workspace.add_analysis(
                    run["total_memory"], workspace.processes_of(run)
                )
                st.rerun()

    processes = workspace.processes_of(run)
    total_memory = run["total_memory"]
    algorithm = st.radio(
        "Algorithm for the next run", ALGORITHM_ORDER, key=f"next::{tab_id}"
    )
    if st.button("Run Another Algorithm", key=f"again::{tab_id}"):
        workspace.add_run(total_memory, processes, algorithm)
        st.rerun()


def show_algorithm_cards(analysis: dict) -> None:
    """One bordered card per algorithm, showing the same facts as the table."""
    for row_start in range(0, len(analysis["rows"]), 2):
        pair = analysis["rows"][row_start : row_start + 2]
        for column, row in zip(st.columns(2), pair):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{row['Algorithm']}**")
                    show_metrics(
                        {
                            "Successful": row["Successful Allocations"],
                            "Failed": row["Failed Allocations"],
                            "Utilization": f"{row['Memory Utilization']}%",
                        }
                    )
                    st.caption(
                        f"Peak {row['allocated']} KB in use at time unit "
                        f"{row['shown_time']}  |  largest free block "
                        f"{row['largest_free_block']} KB  |  "
                        f"{row['free_blocks']} free block(s)  |  "
                        f"{row['Free Memory']} KB free"
                    )
                    st.markdown(row["reject_note"])


def render_analysis(analysis: dict) -> None:
    """An Analysis tab: the four algorithms compared on one shared workload."""
    st.subheader("Algorithm Comparison")
    table = pd.DataFrame(analysis["rows"])[COMPARISON_COLUMNS]
    st.table(table)
    st.bar_chart(table, x="Algorithm", y="Memory Utilization", color="Algorithm")

    if st.button("View Analysis", type="primary"):
        st.subheader("Results")
        show_algorithm_cards(analysis)

        st.markdown("**Comparison summary**")
        st.code("\n".join(workspace.summary_lines(analysis)), language=None)
        st.caption(workspace.verdict(analysis))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Memory Allocation Simulator")
st.write(
    "Simulation of dynamic memory allocation techniques in Operating Systems."
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
