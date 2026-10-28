"""Kernel bake-off on the headline 1y Snowball (spec D11, AC9).

Kernel A (CTMC) with both exact-exponential backends and kernel B (Gauss +
Richardson) are run on a grid ladder against the D12 reference; each row
records base-solve time (median of ``--repeats``, one process window, arms
alternated), PV/delta/gamma error, the engine's identity residuals, or the
refusal. The decision matrix picks the cheapest configuration of each kernel
that meets the budget on PV, delta and gamma.

    python -m example.case_studies.localvol_transition_quad.bakeoff --reference <reference.json> [--repeats 3]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

from .contract import snowball_contract
from .engine import LocalVolQuad
from .grid import GridConfig, UnresolvedGrid
from .kernels import CTMCKernel, GaussRichardsonKernel
from .market import load_fixture
from .reference import HALF_WIDTH_STD, OUT_DIR, budget, source_hashes

ARMS = {
    "ctmc_contour": (lambda: CTMCKernel(backend="contour"), [(200, 1), (300, 1), (400, 1), (200, 3), (300, 3)]),
    "ctmc_chebyshev": (lambda: CTMCKernel(backend="chebyshev"), [(200, 1), (300, 1), (400, 1), (200, 3)]),
    "gauss_richardson": (lambda: GaussRichardsonKernel(1), [(400, 1), (400, 3), (300, 9), (400, 9)]),
}
INDEPENDENCE = {
    "ctmc_contour": "shares the PDE's semi-discrete generator family (finite differences in ln S); "
                    "differs in time (exact exponential, no stepping) and in grid/event placement",
    "ctmc_chebyshev": "as ctmc_contour",
    "gauss_richardson": "integral kernel; shares no spatial operator with the PDE (most independent)",
}
BASE = ("pv", "delta", "gamma")


def run_row(arm: str, cells: int, subdivide: int, contract, market, reference: dict, repeats: int) -> dict:
    factory, _ = ARMS[arm]
    row = dict(arm=arm, cells_base=cells, subdivide=subdivide)
    times = []
    try:
        for _ in range(repeats):
            engine = LocalVolQuad(factory(), GridConfig(cells=cells, half_width_std=HALF_WIDTH_STD,
                                                        subdivide=subdivide))
            start = perf_counter()
            out = engine.price(contract, market)
            times.append(perf_counter() - start)
    except UnresolvedGrid as exc:
        row.update(status="refused", reason=str(exc))
        return row
    row.update(status="priced", seconds=statistics.median(times), seconds_all=times, cells=out["cells"],
               values=out["values"], diagnostics=out["diagnostics"])
    row["error"] = {k: out["values"][k] - reference[k] for k in BASE}
    row["meets_budget"] = all(abs(row["error"][k]) <= budget(k, reference[k]) for k in BASE)
    return row


def decide(rows: list[dict]) -> dict:
    matrix = {}
    for arm in ARMS:
        ok = [r for r in rows if r["arm"] == arm and r.get("meets_budget")]
        best = min(ok, key=lambda r: r["seconds"]) if ok else None
        refused = sum(r["arm"] == arm and r["status"] == "refused" for r in rows)
        matrix[arm] = dict(cheapest_meeting=None if best is None else
                           {k: best[k] for k in ("cells_base", "subdivide", "cells", "seconds", "error")},
                           refused_configs=refused, independence=INDEPENDENCE[arm])
    meeting = {a: m["cheapest_meeting"]["seconds"] for a, m in matrix.items() if m["cheapest_meeting"]}
    winner = min(meeting, key=meeting.get) if meeting else None
    return dict(matrix=matrix, winner=winner)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=str(OUT_DIR / "reference.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default=str(OUT_DIR / "bakeoff.json"))
    args = parser.parse_args(argv)
    ref = json.loads(Path(args.reference).read_text())["cases"]["snowball_1y"]["assessment"]["reference"]
    market = load_fixture()
    contract = snowball_contract(market.spot, 1.0, 12)
    rows = []
    ladder = max(len(cfgs) for _, cfgs in ARMS.values())
    for i in range(ladder):                      # interleave arms rung by rung (one window, alternated)
        order = list(ARMS) if i % 2 == 0 else list(reversed(ARMS))
        for arm in order:
            cfgs = ARMS[arm][1]
            if i < len(cfgs):
                rows.append(run_row(arm, *cfgs[i], contract, market, ref, args.repeats))
                r = rows[-1]
                print(arm, cfgs[i], r["status"], f"{r.get('seconds', float('nan')):.3f}s", r.get("error"), flush=True)
    result = dict(reference=ref, rows=rows, decision=decide(rows), sources=source_hashes())
    Path(args.out).write_text(json.dumps(result, indent=1))
    print("winner:", result["decision"]["winner"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
