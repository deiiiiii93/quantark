"""The PDE timing gate (spec 2 §9.5): cold and warm price medians and peak memory per family.

Run the same file against two source trees back to back, in one window:

    PYTHONPATH=<base tree> python test/pde_bitwise/timing.py --out base.json --reps 5
    PYTHONPATH=<branch tree> python test/pde_bitwise/timing.py --out branch.json --reps 5
    python test/pde_bitwise/timing.py --compare base.json branch.json

cold = a new solver and ``price`` after clearing the module-level market memos;
warm = the same solver, a second ``price``. One warm-up, then ``--reps`` repetitions;
medians are written. Peak memory is ``ru_maxrss`` of a subprocess that prices one family.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import capture_book as cb  # noqa: E402

FAMILIES = ("european", "american", "barrier", "snowball", "phoenix", "ko_reset", "lv_snowball",
            "statistics")


def _case(family):
    """(solver factory, call(solver)) for a family's book-standard flat product."""
    from quantark.asset.equity.engine import pde
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver
    from quantark.asset.equity.param import PDEParams
    from quantark.util.enum import ObservationType

    flat = cb.env_flat()
    table = {
        "european": (pde.EuropeanPDESolver, cb.european(), flat),
        "american": (pde.AmericanPDESolver, cb.american_tenor(), flat),
        "barrier": (pde.BarrierPDESolver, cb.barrier(ObservationType.CONTINUOUS), flat),
        "snowball": (pde.SnowballPDESolver, cb.snowball(), flat),
        "phoenix": (pde.PhoenixPDESolver, cb.make_phoenix(cb.PHOENIX), flat),
        "ko_reset": (pde.KOResetSnowballPDESolver, cb.make_ko_reset(cb.KO_RESET), flat),
        "lv_snowball": (LocalVolSnowballPDESolver, cb.snowball(), cb.env_grid_vol()),
        "statistics": (pde.SnowballPDESolver, cb.snowball(), flat),
    }
    cls, product, env = table[family]
    if family == "statistics":
        return (lambda: cls(params=PDEParams())), (lambda s: s.calculate_event_stats(product, env))
    return (lambda: cls(params=PDEParams())), (lambda s: s.price(product, env))


def _clear_memos():
    from quantark.asset.equity.engine.pde import base_pde_solver

    # both module memos exist on 0.6.0 and 0.7.0: a cold solve starts with neither
    for name in ("_ENV_STEP_COEFF_MEMO", "_ENV_DF_MEMO"):
        memo = getattr(base_pde_solver, name, None)
        if memo is not None:
            memo.clear()


def _once(family):
    make, call = _case(family)
    _clear_memos()
    t0 = time.perf_counter()
    solver = make()
    call(solver)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    call(solver)
    return cold, time.perf_counter() - t0


def _peak_rss(family) -> int:
    """ru_maxrss (bytes on macOS, KiB on Linux) of a fresh process pricing one family."""
    out = subprocess.run([sys.executable, __file__, "--memory-family", family],
                         capture_output=True, text=True, env=os.environ.copy(), check=True)
    return int(out.stdout.strip().splitlines()[-1])


def measure(reps: int) -> dict:
    result = {}
    for family in FAMILIES:
        _once(family)                                    # warm-up
        cold, warm = zip(*(_once(family) for _ in range(reps)))
        result[family] = {"cold": statistics.median(cold), "warm": statistics.median(warm),
                          "peak_rss": _peak_rss(family)}
    return result


def compare(base_path, branch_path, limit=0.10) -> int:
    base, branch = (json.loads(Path(p).read_text()) for p in (base_path, branch_path))
    worst = 0
    for family in FAMILIES:
        b, n = base[family], branch[family]
        row = []
        for key in ("cold", "warm", "peak_rss"):
            ratio = n[key] / b[key]
            row.append(f"{key} {ratio:5.3f}")
            if ratio > 1.0 + limit:
                worst = 1
        print(f"{family:12s} " + "  ".join(row))
    print("PASS" if worst == 0 else f"REGRESSION (> {limit:.0%})")
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--memory-family")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args(argv)
    if args.memory_family:
        _once(args.memory_family)
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return 0
    if args.compare:
        return compare(*args.compare)
    Path(args.out).write_text(json.dumps(measure(args.reps), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
