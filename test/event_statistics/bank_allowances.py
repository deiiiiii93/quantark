"""Bank the grid engines' discretisation allowances of the fast cross-engine smoke (spec AC20, D16; T22).

For every case x grid engine of ``cases.CASES``: the statistics at the three levels ``L0`` (the smoke's), ``L1`` and
``L2`` (refined twice), and per compared number

- ``L0``: the value the smoke reproduces live (``test_cross_engine_smoke`` checks it within ``golden_compare``
  tolerances, so a banked allowance always belongs to the engine that runs);
- ``allowance``: ``max(|L0 - L1|, |L0 - L2|)``, the discretisation allowance the MC gate adds to its k x SE (the
  envelope of L0's distance to both refined levels: per number the convergence is not monotone on every engine --
  PDE 1D and DCN PDE presets move a number further from L1 to L2 than from L0 to L1 -- so ``|L0 - L2|`` alone
  could understate L0's error);
- ``refinement``: ``[max|L0 - L1|, max|L1 - L2|]``, the successive refinement steps (recorded, not asserted:
  see above);
- ``residuals``: ``probability_residual`` and ``reconciliation_error / max(1, |pv|)`` at each level (spec AC8, D16:
  the bounds' refinement evidence).

Run from the worktree root (banking interpreter: the main ``.venv``, py3.11)::

    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python test/event_statistics/bank_allowances.py

Never edit ``allowances.json`` by hand; re-bank only for a deliberate engine change, with a field-level diff (D22).
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from event_statistics.cases import CASES, LEVELS, fields_of, grid_statistics  # noqa: E402

OUT = HERE / "allowances.json"


def main() -> None:
    bank = {"_provenance": {"python": platform.python_version(), "machine": platform.machine(),
                            "generated": time.strftime("%Y-%m-%d"), "script": "test/event_statistics/bank_allowances.py",
                            "levels": "L0 = the smoke's engine; L1, L2 refined (cases.py)"}}
    for name, case in CASES.items():
        bank[name] = {}
        for engine in case.grid:
            levels = {}
            seconds = {}
            residuals = {}
            for level in LEVELS:
                start = time.perf_counter()
                stats = grid_statistics(case, engine, level)
                seconds[level] = round(time.perf_counter() - start, 3)
                levels[level] = fields_of(stats)
                residuals[level] = [float(stats.probability_residual),
                                    float(stats.reconciliation_error) / max(1.0, abs(float(stats.pv)))]
            record = {"L0": {}, "allowance": {}, "refinement": {}, "seconds": seconds, "residuals": residuals}
            for path, value in levels["L0"].items():
                record["L0"][path] = value.tolist()
                if path == "ki_table.times":
                    continue
                gap01 = np.abs(value - levels["L1"][path])
                gap02 = np.abs(value - levels["L2"][path])
                gap12 = np.abs(levels["L1"][path] - levels["L2"][path])
                record["allowance"][path] = np.maximum(gap01, gap02).tolist()
                record["refinement"][path] = [float(gap01.max()), float(gap12.max())]
            bank[name][engine] = record
            print(f"{name:34s} {engine:24s} {seconds}", flush=True)
    OUT.write_text(json.dumps(bank, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
