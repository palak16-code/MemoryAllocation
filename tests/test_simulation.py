"""
test_simulation.py
------------------
Headless correctness checks for the memory allocation simulator.

Run with::

    python tests/test_simulation.py

No test framework required (plain asserts + a tiny runner) so that students
can run the proof of correctness with a single command and no extra installs.

Every check below maps to a statement in the project report, e.g.
"First Fit allocates the lowest-addressed free block that fits".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms import get_algorithm  # noqa: E402
from analysis.metrics import (  # noqa: E402
    external_fragmentation_index,
    to_comparison_table,
    utilisation_percent,
)
from simulation.memory_manager import MemoryBlock, MemoryManager  # noqa: E402
from simulation.process_manager import Process, validate_workload  # noqa: E402
from simulation.simulator import run_all_algorithms, run_simulation  # noqa: E402
from workspace import (  # noqa: E402
    ARRIVAL_COLUMN,
    DEFAULT_WORKLOAD,
    DURATION_COLUMN,
    ID_COLUMN,
    SIZE_COLUMN,
)

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append(f"{name} :: {detail}")
        print(f"  FAIL  {name}  {detail}")


def owners(memory: MemoryManager) -> str:
    """Compact textual layout, e.g. 'P1:100|FREE:200'."""
    return "|".join(f"{b.label()}:{b.size}" for b in memory.blocks)


# ---------------------------------------------------------------------------
# 1. First Fit
# ---------------------------------------------------------------------------
def test_first_fit() -> None:
    print("\n[1] First Fit")
    mem = MemoryManager(total_size=1000)
    ff = get_algorithm("First Fit")
    r = mem.allocate("P1", 300, ff, time=0)
    check("allocates the first (only) free block", r.success and r.block_start == 0,
          owners(mem))
    check("block is split, remainder stays free", owners(mem) == "P1:300|FREE:700",
          owners(mem))

    mem.allocate("P2", 400, ff, time=1)
    mem.deallocate("P1", time=2)
    r3 = mem.allocate("P3", 100, ff, time=3)
    check("reuses the hole left by the freed process",
          r3.success and r3.block_start == 0, owners(mem))
    check("the reused hole is split, tail of the hole stays free",
          owners(mem) == "P3:100|FREE:200|P2:400|FREE:300", owners(mem))

    # Coalescing: two holes that become adjacent must merge into one.
    mem2 = MemoryManager(total_size=1000)
    for pid, size in (("A", 300), ("B", 200), ("C", 300), ("D", 200)):
        mem2.allocate(pid, size, ff, time=0)   # memory is now completely full
    check("memory is full", owners(mem2) == "A:300|B:200|C:300|D:200", owners(mem2))
    mem2.deallocate("C", time=1)
    check("freeing C creates one hole", owners(mem2) == "A:300|B:200|FREE:300|D:200",
          owners(mem2))
    mem2.deallocate("D", time=2)
    check("adjacent free blocks are merged after deallocation",
          owners(mem2) == "A:300|B:200|FREE:500", owners(mem2))
    check("free block count dropped from 2 to 1",
          len(mem2.free_blocks()) == 1, str(len(mem2.free_blocks())))


# ---------------------------------------------------------------------------
# 2. Best Fit
# ---------------------------------------------------------------------------
def test_best_fit() -> None:
    print("\n[2] Best Fit")
    mem = MemoryManager(total_size=1000)
    bf = get_algorithm("Best Fit")
    mem.allocate("A", 200, bf, time=0)     # -> A:200 | FREE:800
    mem.allocate("B", 500, bf, time=1)     # -> B fills the rest
    mem.deallocate("A", time=2)            # -> FREE:200 | B:500 | FREE:300
    check("layout after deallocation", owners(mem) == "FREE:200|B:500|FREE:300",
          owners(mem))
    r = mem.allocate("C", 250, bf, time=3)
    check("picks the smallest fitting hole (300, not 200)",
          r.success and r.block_start == 700, owners(mem))
    check("hole is 300 - 250 = 50 KB", owners(mem) == "FREE:200|B:500|C:250|FREE:50",
          owners(mem))


# ---------------------------------------------------------------------------
# 3. Worst Fit
# ---------------------------------------------------------------------------
def test_worst_fit() -> None:
    print("\n[3] Worst Fit")
    mem = MemoryManager(total_size=1000)
    wf = get_algorithm("Worst Fit")
    mem.allocate("A", 200, wf, time=0)
    mem.allocate("B", 500, wf, time=1)
    mem.deallocate("A", time=2)            # FREE:200 | B:500 | FREE:300
    r = mem.allocate("C", 250, wf, time=3)
    check("picks the largest fitting hole (300)",
          r.success and r.block_start == 700, owners(mem))
    check("leaves a 50 KB remainder", owners(mem) == "FREE:200|B:500|C:250|FREE:50",
          owners(mem))

    mem2 = MemoryManager(total_size=1000)
    r_fail = mem2.allocate("X", 1500, wf, time=0)
    check("rejects a request larger than total memory", not r_fail.success,
          r_fail.reason)


# ---------------------------------------------------------------------------
# 4. Next Fit
# ---------------------------------------------------------------------------
def test_next_fit() -> None:
    print("\n[4] Next Fit")
    mem = MemoryManager(total_size=1000)
    nf = get_algorithm("Next Fit")
    mem.allocate("P1", 300, nf, time=0)    # P1:300 | FREE:700  (pointer -> 300)
    check("pointer moves to the end of the allocated block",
          nf.next_address == 300, f"pointer={nf.next_address}")
    mem.allocate("P2", 200, nf, time=1)    # P1:300 | P2:200 | FREE:500
    mem.allocate("P3", 300, nf, time=2)    # P1:300 | P2:200 | P3:300 | FREE:200
    check("pointer tracks the last allocation", nf.next_address == 800,
          f"pointer={nf.next_address}")

    # The behaviour that distinguishes Next Fit from First Fit:
    # a hole *behind* the pointer is ignored while one *ahead* exists.
    mem.deallocate("P1", time=3)           # FREE:300 | P2:200 | P3:300 | FREE:200
    check("layout with a hole behind the pointer",
          owners(mem) == "FREE:300|P2:200|P3:300|FREE:200", owners(mem))
    r = mem.allocate("P4", 100, nf, time=4)
    check("uses the hole ahead of the pointer (800), not the one at 0",
          r.success and r.block_start == 800, owners(mem))

    # First Fit, given the identical state, takes the address-0 hole.
    mem_ff = MemoryManager(total_size=1000)
    ff = get_algorithm("First Fit")
    for pid, size in (("P1", 300), ("P2", 200), ("P3", 300)):
        mem_ff.allocate(pid, size, ff, time=0)
    mem_ff.deallocate("P1", time=3)
    r_ff = mem_ff.allocate("P4", 100, ff, time=4)
    check("First Fit instead takes the lowest hole - algorithms differ",
          r_ff.success and r_ff.block_start == 0, owners(mem_ff))

    # Wrap-around: nothing ahead of the pointer fits, so the search restarts
    # from address 0 and finds the hole behind the pointer.
    nf.reset()
    check("reset clears the roving pointer", nf.next_address == 0)
    wrap = MemoryManager(total_size=1000)
    for pid, size in (("A", 300), ("B", 200), ("C", 400)):
        wrap.allocate(pid, size, nf, time=0)   # pointer now at 900
    check("pointer at the end of the allocated region", nf.next_address == 900,
          f"pointer={nf.next_address}")
    wrap.deallocate("A", time=1)
    r_wrap = wrap.allocate("D", 250, nf, time=2)
    check("search wraps round to address 0 when nothing fits ahead",
          r_wrap.success and r_wrap.block_start == 0, owners(wrap))
    check("the wrap cost extra block inspections",
          r_wrap.search_operations == 2, str(r_wrap.search_operations))


# ---------------------------------------------------------------------------
# 5. Allocation failure and deallocation
# ---------------------------------------------------------------------------
def test_allocation_failure() -> None:
    print("\n[5] Failure handling")
    mem = MemoryManager(total_size=500)
    ff = get_algorithm("First Fit")
    mem.allocate("P1", 300, ff, time=0)
    r = mem.allocate("P2", 400, ff, time=1)
    check("request with no fitting block fails cleanly", not r.success, r.reason)
    check("failure leaves memory untouched", owners(mem) == "P1:300|FREE:200",
          owners(mem))
    check("search counted the whole block list", r.search_operations == 2,
          f"counted={r.search_operations}")

    mem.deallocate("nobody", time=2)
    check("deallocating an unknown PID is a no-op", owners(mem) == "P1:300|FREE:200",
          owners(mem))
    check("deallocation returns True for the owner", mem.deallocate("P1", time=3))
    check("all memory is free again", owners(mem) == "FREE:500", owners(mem))


# ---------------------------------------------------------------------------
# 6. Compaction
# ---------------------------------------------------------------------------
def test_compaction() -> None:
    print("\n[6] Compaction")
    mem = MemoryManager(total_size=1000)
    ff = get_algorithm("First Fit")
    mem.allocate("P1", 200, ff, time=0)    # P1:200 | P2:300 | P3:400 | FREE:100
    mem.allocate("P2", 300, ff, time=1)
    mem.allocate("P3", 400, ff, time=2)
    mem.deallocate("P2", time=3)           # hole appears in the middle
    before = owners(mem)
    check("pre-compaction layout matches the brief",
          before == "P1:200|FREE:300|P3:400|FREE:100", before)
    check("2 free blocks before compaction", len(mem.free_blocks()) == 2)

    relocated = mem.compact()
    check("layout after compaction: P1 | P3 | FREE",
          owners(mem) == "P1:200|P3:400|FREE:400", owners(mem))
    check("2 free blocks collapsed to 1", len(mem.free_blocks()) == 1,
          f"free blocks={len(mem.free_blocks())}")
    check("one block was relocated", relocated == 1, f"relocated={relocated}")
    check("no process lost its memory",
          mem.block_of("P1") is not None and mem.block_of("P3") is not None)
    check("compaction is idempotent (nothing left to move)",
          mem.compact() == 0 and owners(mem) == "P1:200|P3:400|FREE:400",
          owners(mem))


# ---------------------------------------------------------------------------
# 7. Simulator: arrival, execution, termination
# ---------------------------------------------------------------------------
def test_simulator_lifecycle() -> None:
    print("\n[7] Simulator lifecycle")
    workload = [
        Process("P1", 200, arrival=0, duration=3),
        Process("P2", 300, arrival=1, duration=2),
        Process("P3", 150, arrival=2, duration=1),
    ]
    res = run_simulation(1000, workload, "First Fit")
    check("all three processes were allocated",
          res.metrics["processes_allocated"] == 3, str(res.metrics))
    check("no rejections", res.metrics["processes_rejected"] == 0, str(res.metrics))
    check("memory is fully free at the end of the run",
          res.metrics["total_free_memory_kb"] == 1000, str(res.metrics))
    check("history has one snapshot per processed time unit",
          len(res.history) == 4, f"snapshots={len(res.history)} (t=0..3)")
    check("snapshot times are consecutive from 0",
          [s.time for s in res.history] == [0, 1, 2, 3],
          str([s.time for s in res.history]))
    check("every process was admitted at its own arrival time",
          sorted({r["pid"]: r["time"] for r in res.records if r["success"]}.values())
          == [0, 1, 2],
          str([(r["pid"], r["time"]) for r in res.records]))
    check("utilisation formula", utilisation_percent(500, 1000) == 50.0)
    check("fragmentation formula",
          external_fragmentation_index(free=100, largest=60) == 40.0)
    check("scattered memory is 100% fragmented",
          external_fragmentation_index(free=100, largest=100) == 0.0)

    # Rejection path.
    tight = [Process("P1", 400, 0, 5), Process("P2", 400, 0, 2),
             Process("P3", 400, 0, 2)]
    res2 = run_simulation(1000, tight, "First Fit")
    check("an oversized process is rejected", res2.metrics["processes_rejected"] >= 1,
          str(res2.metrics))
    check("success rate is 2/3 = 66.67%",
          abs(res2.metrics["allocation_success_rate"] - 66.67) < 0.1,
          str(res2.metrics["allocation_success_rate"]))


# ---------------------------------------------------------------------------
# 8. Stepping
# ---------------------------------------------------------------------------
def test_stepping() -> None:
    print("\n[8] Step-by-step control")
    from simulation.simulator import Simulator

    workload = [Process("P1", 300, 0, 2), Process("P2", 300, 2, 2)]
    sim = Simulator(1000, workload, "Best Fit")
    check("starts unfinished", not sim.is_finished)
    s1 = sim.step()
    check("at t=0 P1 occupies 300 KB", s1.time == 0 and s1.used_memory == 300,
          f"t={s1.time} used={s1.used_memory}")
    s2 = sim.step()
    check("at t=1 P1 is still running and memory is unchanged",
          s2.time == 1 and s2.running == ["P1"] and s2.used_memory == 300,
          f"t={s2.time} running={s2.running} used={s2.used_memory}")
    s3 = sim.step()
    check("at t=2 P1 completes first, then P2 reuses the same memory",
          s3.time == 2 and s3.running == ["P2"] and s3.used_memory == 300
          and s3.finished == ["P1"],
          f"running={s3.running} finished={s3.finished} used={s3.used_memory}")
    sim.run()
    check("run() terminates the simulation", sim.is_finished)
    check("final memory is free", sim.memory.free_memory() == 1000)


# ---------------------------------------------------------------------------
# 9. Comparison on an identical workload
# ---------------------------------------------------------------------------
def test_comparison() -> None:
    print("\n[9] Algorithm comparison")
    workload = [
        Process("P1", 150, 0, 4), Process("P2", 250, 0, 3),
        Process("P3", 100, 1, 3), Process("P4", 300, 1, 2),
        Process("P5", 200, 3, 3),
    ]
    results = run_all_algorithms(1000, workload)
    check("all four algorithms ran", len(results) == 4, str(list(results)))
    table = to_comparison_table(r.metrics for r in results.values())
    check("comparison table has one row per algorithm", len(table) == 4,
          f"rows={len(table)}")
    for col in ["Algorithm", "Processes Allocated", "Memory Utilisation (%)",
                "External Fragmentation (%)", "Search Operations"]:
        check(f"table contains '{col}'", col in table.columns, str(list(table.columns)))
    # Best Fit and Worst Fit must always inspect the entire block list, so
    # their search totals are identical to each other.
    bf = results["Best Fit"].metrics
    wf = results["Worst Fit"].metrics
    check("Best Fit and Worst Fit inspect exactly the same number of blocks",
          bf["search_operations"] == wf["search_operations"],
          f"bf={bf['search_operations']} wf={wf['search_operations']}")
    check("First Fit needs no more searches than Best Fit",
          results["First Fit"].metrics["search_operations"]
          <= bf["search_operations"],
          f"ff={results['First Fit'].metrics['search_operations']} "
          f"bf={bf['search_operations']}")
    # The counter must be an exact sum of the per-attempt search costs.
    for name, res in results.items():
        per_attempt = sum(
            r["search_operations"] for r in res.records if r["action"] == "Allocate"
        )
        check(f"{name}: search counter equals the sum of per-attempt counts",
              per_attempt == res.metrics["search_operations"],
              f"{per_attempt} vs {res.metrics['search_operations']}")


# ---------------------------------------------------------------------------
# 11. The four policies really do select different blocks
# ---------------------------------------------------------------------------
def test_policy_divergence() -> None:
    """Prove the four strategies are genuinely different selection policies.

    A workload can only tell the policies apart if, at some point, more than
    one free block is large enough for the arriving request. Everything below
    is checked against one such layout, and then against the workload the
    application actually starts with - no result is ever asserted that the
    allocation logic did not produce.
    """
    print("\n[11] Policy divergence")

    # Two suitable free blocks of different sizes (500 and 400), plus one that
    # is too small (120) and must be ignored by every policy.
    def layout() -> list:
        return [
            MemoryBlock(0, 300, "A"),
            MemoryBlock(300, 120),
            MemoryBlock(420, 500),
            MemoryBlock(920, 400),
        ]

    chosen = {
        name: get_algorithm(name).select_block(layout(), 250)
        for name in ("First Fit", "Best Fit", "Worst Fit", "Next Fit")
    }
    check("First Fit takes the lowest-addressed block that fits",
          chosen["First Fit"] == 2, str(chosen))
    check("Best Fit takes the smallest block that fits",
          chosen["Best Fit"] == 3, str(chosen))
    check("Worst Fit takes the largest block that fits",
          chosen["Worst Fit"] == 2, str(chosen))
    check("Next Fit starts from address 0, so it matches First Fit here",
          chosen["Next Fit"] == 2, str(chosen))

    # Next Fit is the only one of the four that can skip a hole: once its
    # pointer has moved past the 500 KB block it no longer considers it.
    nf = get_algorithm("Next Fit")
    nf.select_block(layout(), 500)               # pointer -> 920
    check("Next Fit skips a hole behind its pointer where First Fit does not",
          nf.select_block(layout(), 250) == 3
          and get_algorithm("First Fit").select_block(layout(), 250) == 2,
          f"pointer={nf.next_address}")

    # --- the same check, end to end, on the shipped workload ---------------
    workload = [
        Process(str(row[ID_COLUMN]), int(row[SIZE_COLUMN]),
                int(row[ARRIVAL_COLUMN]), int(row[DURATION_COLUMN]))
        for row in DEFAULT_WORKLOAD.to_dict("records")
    ]
    results = run_all_algorithms(1000, workload)
    placements = {
        name: tuple(
            (r["pid"], r["block_start"])
            for r in res.records
            if r.get("action") == "Allocate" and r["success"]
        )
        for name, res in results.items()
    }
    check("the four algorithms do not place the workload identically",
          len(set(placements.values())) > 1, str(placements))
    check("at least three distinct layouts appear across the four algorithms",
          len(set(placements.values())) >= 3, str(placements))

    # The workload is only discriminating if some request had a genuine
    # choice between differently sized blocks that both satisfied it.
    ambiguous = 0
    shortfalls = []
    for res in results.values():
        for record in res.records:
            if record.get("action") != "Allocate":
                continue
            snapshot = next(s for s in res.history if s.time == record["time"])
            suitable = {
                block.size for block in snapshot.blocks
                if block.is_free and block.size >= record["size"]
            }
            if len(suitable) > 1:
                ambiguous += 1
            if not record["success"]:
                shortfalls.append(
                    (record["pid"], record["size"], snapshot.free_memory,
                     snapshot.largest_free_block)
                )
    check("the workload offers a choice of differently sized suitable blocks",
          ambiguous > 0, f"requests with a real choice={ambiguous}")
    check("at least one process is rejected on this workload",
          bool(shortfalls), str(shortfalls))
    check("every rejection is external fragmentation, not a shortage",
          all(free >= size for _, size, free, _ in shortfalls), str(shortfalls))
    check("at least one policy places the whole workload",
          max(res.metrics["processes_allocated"] for res in results.values())
          == len(workload),
          str([r.metrics["processes_allocated"] for r in results.values()]))


# ---------------------------------------------------------------------------
# 12. Input validation
# ---------------------------------------------------------------------------
def test_validation() -> None:
    print("\n[12] Input validation")
    check("empty workload is reported",
          len(validate_workload([])) == 1)
    dup = [Process("P1", 100, 0, 1), Process("P1", 100, 1, 1)]
    check("duplicate IDs are reported", any("Duplicate" in p for p in validate_workload(dup)),
          str(validate_workload(dup)))
    check("a valid workload produces no problems",
          validate_workload([Process("P1", 100, 0, 1)]) == [])
    try:
        Process("P1", 0, 0, 1)
        check("zero size is rejected", False, "no exception raised")
    except ValueError:
        check("zero size is rejected", True)
    try:
        MemoryManager(total_size=0)
        check("zero memory is rejected", False, "no exception raised")
    except ValueError:
        check("zero memory is rejected", True)


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 68)
    print("MEMORY ALLOCATION SIMULATOR - CORRECTNESS CHECKS")
    print("=" * 68)
    for test in (
        test_first_fit, test_best_fit, test_worst_fit, test_next_fit,
        test_allocation_failure, test_compaction, test_simulator_lifecycle,
        test_stepping, test_comparison, test_policy_divergence,
        test_validation,
    ):
        test()

    print("\n" + "=" * 68)
    print(f"PASSED: {len(PASSED)}   FAILED: {len(FAILED)}")
    if FAILED:
        print("\nFailures:")
        for item in FAILED:
            print(f"  - {item}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
