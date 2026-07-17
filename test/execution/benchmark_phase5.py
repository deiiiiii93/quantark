"""Phase 5 scenario-parallelism benchmark (dev-machine evidence).

Full solution-shaped surface-shock menu (4 global + 2 tenor + 3 moneyness
= 9 cells on the synthetic fixture) priced serially and on 4 spawn
processes, with the complete-payload equality check on EVERY repetition —
a speedup with a payload mismatch is a FAIL.

Spec section 20 gate 6 — "independent scenario grids with serial wall
time above ten seconds achieve at least 2.5x on four processes" — is a
CONTROLLED-HOST production gate on production-sized cells; this script is
dev-machine attribution evidence only (kickoff decision 2026-07-17,
matching the Phase 2/4 treatment).

Dev-machine snapshot (Apple Silicon macOS, 2026-07-17, paths=2^12,
median of 3):

    serial      9 cells   3.26 s
    processes4  9 cells   2.07 s   -> x1.58, 194/194 fields match per rep

Attribution: fixture cells are TINY (~0.36 s each), so pool startup and
the per-worker quantark import are a large fixed cost, and the single
heaviest cell (heston recalibrate, ~1s of calibration x2) lower-bounds
the parallel wall. The spec gate targets grids whose SERIAL time exceeds
10 s, where the fixed costs amortize away — hence controlled-host,
production-sized cells for the release gate.

Run:  PYTHONPATH=$PWD .venv/bin/python test/execution/benchmark_phase5.py
"""
import dataclasses
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # test/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))         # execution/

PATHS = 2 ** 12
REPS = 3
WORKERS = 4


def _base_ref():
    from quantark.execution.scenario.contracts import BaseInputsRef
    from surface_shock_process_helpers import SURFACE_SHOCK_TEST_FACTORY_ID

    return BaseInputsRef(
        factory_id=SURFACE_SHOCK_TEST_FACTORY_ID,
        payload=(("num_paths", PATHS), ("seed", 42)),
    )


def _specs():
    from quantark.asset.equity.riskmeasures.surface_shock_scenarios import (
        build_surface_shock_cells,
        cells_to_scenario_specs,
    )

    cells = build_surface_shock_cells(
        tenors=(91 / 365.0, 182 / 365.0),
        moneyness_buckets=((-0.40, -0.10), (-0.10, 0.10), (0.10, 0.40)),
        dsigma=0.005,
    )
    return cells_to_scenario_specs(cells)


def _context(backend, workers):
    from quantark.execution.context import default_context
    from quantark.execution.policy import (
        ExecutionPolicy,
        ExecutorSelection,
        ResourceBudget,
    )

    return dataclasses.replace(
        default_context(),
        execution_policy=ExecutionPolicy(
            scenario=ExecutorSelection(backend=backend, workers=workers),
        ),
        resource_budget=ResourceBudget(max_processes=workers, max_threads=1),
    )


def _run(backend, workers, specs):
    from quantark.execution.api import PricingSession

    start = time.perf_counter()
    with PricingSession(_context(backend, workers)) as session:
        outcomes = session.run_scenarios(_base_ref(), specs, None)
    return time.perf_counter() - start, outcomes


def main() -> None:
    from quantark.execution.scenario.validate import compare_scenario_outcomes

    specs = _specs()
    print(f"cells={len(specs)} paths={PATHS} reps={REPS} workers={WORKERS}")

    serial_times, process_times = [], []
    for rep in range(REPS):
        serial_seconds, serial_outcomes = _run("serial", 1, specs)
        process_seconds, process_outcomes = _run("processes", WORKERS, specs)
        report = compare_scenario_outcomes(serial_outcomes, process_outcomes)
        status = "MATCH" if report.all_scenarios_match else (
            f"MISMATCH at {report.first_mismatch_path}"
        )
        print(
            f"rep{rep}: serial {serial_seconds:6.2f}s  "
            f"processes{WORKERS} {process_seconds:6.2f}s  "
            f"x{serial_seconds / process_seconds:4.2f}  "
            f"fields {report.fields_matching}/{report.fields_compared}  "
            f"{status}"
        )
        if not report.all_scenarios_match:
            raise SystemExit("payload mismatch: benchmark is INVALID")
        serial_times.append(serial_seconds)
        process_times.append(process_seconds)

    serial_median = statistics.median(serial_times)
    process_median = statistics.median(process_times)
    print(
        f"median: serial {serial_median:.2f}s, processes{WORKERS} "
        f"{process_median:.2f}s -> x{serial_median / process_median:.2f} "
        "(dev machine; the >=2.5x release gate runs on the controlled "
        "host with production-sized cells, spec section 20)"
    )


if __name__ == "__main__":
    main()
