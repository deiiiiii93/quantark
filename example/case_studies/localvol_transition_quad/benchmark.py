"""Matched-accuracy tuning (spec D13, T7) and the cold/warm full-risk benchmark (spec D14, T8).

    python -m example.case_studies.localvol_transition_quad.benchmark tune [--cases ...] [--engines ...]
    python -m example.case_studies.localvol_transition_quad.benchmark time [--repeats 3]

``tune`` runs a fixed configuration ladder per engine and case against the D12
reference and keeps the cheapest configuration whose WORST output is within
budget (the binding quantity is recorded). ``time`` re-times the tuned
configurations: an untimed process warm-up, then per repeat a COLD workload
(fresh engine objects, empty operator caches) for every engine with the order
alternated, plus a ``warm_moved`` row at s0*(1+0.5%) on retained QUAD kernels
(re-anchored grid, so no operator cache can hit). Identical-input replay is not
timed. Run with OPENBLAS/OMP/VECLIB_*_NUM_THREADS=1 on a quiet machine.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from quantark.asset.equity.engine.pde.grid.config import GridConfig as PDEGrid
from quantark.asset.equity.param import PDEParams

from . import adapters
from .engine import OUTPUTS, LocalVolQuad
from .grid import GridConfig, UnresolvedGrid
from .kernels import CTMCKernel
from .market import load_fixture
from .reference import OUT_DIR, budget, cases, source_hashes

QUAD_LADDER = [(c, hw) for hw in (6.0, 8.0) for c in (200, 250, 300, 350, 400, 500, 600)]
PDE_LADDER = [(100, 1), (150, 1), (200, 1), (150, 2), (200, 2), (300, 2), (200, 4), (300, 4), (400, 4)]
# Measured at 2^15 paths: gamma SE ~1-2 (budget 0.035) and theta SE ~1.8e-4 (budget 1e-5) need ~2^28 / ~2^25
# paths, beyond D13's 2^22 cap; the ladder stops at 2^20 and the report extrapolates the paths required.
MC_LADDER = [(1, 2 ** 16), (1, 2 ** 18), (1, 2 ** 20), (4, 2 ** 16)]
MC_BATCH = 2 ** 14
MOVE = 0.005


def quad_engine(cells, half_width):
    return LocalVolQuad(CTMCKernel(), GridConfig(cells=cells, half_width_std=half_width))


def pde_params(points, spd):
    return PDEParams(accuracy="fast", grid=PDEGrid(points=points, steps_per_day=spd))


def mc_config(substeps, paths):
    return adapters.MCConfig(total_paths=paths, batches=max(4, paths // MC_BATCH), substeps=substeps)


def full_risk(engine: str, config, contract, market) -> dict:
    if engine == "quad":
        return quad_engine(*config).full_risk(contract, market)
    if engine == "pde":
        return adapters.pde_full_risk(contract, market, pde_params(*config))
    return adapters.mc_full_risk(contract, market, mc_config(*config))


def score(values: dict, ref: dict, stderr: dict | None = None) -> dict:
    """Worst error as a fraction of budget; MC adds 2 SE (spec D13)."""
    ratio = {k: (abs(values[k] - ref[k]) + (2 * stderr[k] if stderr else 0.0)) / budget(k, ref[k]) for k in OUTPUTS}
    binding = max(ratio, key=ratio.get)
    return dict(ratio=ratio, worst=ratio[binding], binding=binding, meets=ratio[binding] <= 1.0)


def ladder(engine: str):
    return {"quad": QUAD_LADDER, "pde": PDE_LADDER, "mc": MC_LADDER}[engine]


def tune(args) -> int:
    path = OUT_DIR / "tuning.json"
    data = json.loads(path.read_text()) if args.resume and path.exists() else {"cases": {}}
    data["sources"] = source_hashes()
    refs = json.loads((OUT_DIR / "reference.json").read_text())["cases"]
    market = load_fixture()
    for name, contract in cases(market).items():
        if args.cases and name not in args.cases:
            continue
        ref = refs[name]["assessment"]["reference"]
        case = data["cases"].setdefault(name, {})
        for engine in args.engines:
            if engine == "mc" and contract.kind != "snowball":
                continue
            rows = case.setdefault(engine, {})
            for config in ladder(engine):
                key = "x".join(str(x) for x in config)
                if key in rows:
                    continue
                start = perf_counter()
                try:
                    out = full_risk(engine, config, contract, market)
                except UnresolvedGrid as exc:
                    rows[key] = dict(config=list(config), status="refused", reason=str(exc))
                else:
                    rows[key] = dict(config=list(config), status="priced", seconds=perf_counter() - start,
                                     values=out["values"], stderr=out.get("stderr"),
                                     **score(out["values"], ref, out.get("stderr")))
                print(name, engine, key, rows[key]["status"], f"{rows[key].get('seconds', 0):.2f}s",
                      f"worst {rows[key].get('worst', float('nan')):.3f} ({rows[key].get('binding')})", flush=True)
                path.write_text(json.dumps(data, indent=1))
            meeting = [r for r in rows.values() if r.get("meets")]
            case[f"{engine}_chosen"] = min(meeting, key=lambda r: r["seconds"])["config"] if meeting else None
            path.write_text(json.dumps(data, indent=1))
    return 0


def _peak_rss() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def time_runs(args) -> int:
    tuning = json.loads((OUT_DIR / "tuning.json").read_text())["cases"]
    refs = json.loads((OUT_DIR / "reference.json").read_text())["cases"]
    market = load_fixture()
    all_cases = cases(market)
    result = dict(sources=source_hashes(), repeats=args.repeats, environment=dict(
        python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, machine=platform.machine(),
        processor=platform.processor(), threads="OPENBLAS/OMP/VECLIB_MAXIMUM_THREADS=1"), cases={})
    # untimed process warm-up: imports, BLAS/LAPACK first calls, surface loading
    warm = all_cases["put_1y"]
    quad_engine(200, 6.0).price(warm, market)
    adapters.pde_full_risk(warm, market, pde_params(100, 1))
    for name in args.cases or list(all_cases):
        contract = all_cases[name]
        chosen = {e: tuning[name].get(f"{e}_chosen") for e in ("quad", "pde", "mc")}
        engines = [e for e in ("quad", "pde", "mc") if chosen[e] is not None]
        if "mc" in engines and not args.with_mc:
            engines.remove("mc")
        rows = {e: dict(config=chosen[e], cold=[], valuations=[]) for e in engines}
        for rep in range(args.repeats):
            order = engines if rep % 2 == 0 else engines[::-1]
            for e in order:
                start = perf_counter()
                out = full_risk(e, tuple(chosen[e]), contract, market)
                rows[e]["cold"].append(perf_counter() - start)
                rows[e]["valuations"].append(out["timings"])
                rows[e]["values"] = out["values"]
                rows[e]["peak_rss_after"] = _peak_rss()
        if "quad" in engines:                   # warm_moved: retained kernel, re-anchored grid
            engine = quad_engine(*chosen["quad"])
            engine.full_risk(contract, market)
            moved_market = replace(market, spot=market.spot * (1 + MOVE))
            moved = replace(contract, s0=market.spot * (1 + MOVE))
            times = []
            for _ in range(args.repeats):
                start = perf_counter()
                engine.full_risk(moved, moved_market)
                times.append(perf_counter() - start)
            rows["quad"]["warm_moved"] = times
        ref = refs[name]["assessment"]["reference"]
        for e, row in rows.items():
            row["median_cold"] = statistics.median(row["cold"])
            row["accuracy"] = score(row["values"], ref)
        if "quad" in rows and "pde" in rows:
            rows["speedup_vs_pde"] = rows["pde"]["median_cold"] / rows["quad"]["median_cold"]
        if "quad" in rows and "mc" in rows:
            rows["speedup_vs_mc"] = rows["mc"]["median_cold"] / rows["quad"]["median_cold"]
        result["cases"][name] = rows
        print(name, {e: round(r["median_cold"], 3) for e, r in rows.items() if isinstance(r, dict)},
              {k: round(v, 2) for k, v in rows.items() if k.startswith("speedup")}, flush=True)
        (OUT_DIR / "benchmark.json").write_text(json.dumps(result, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tune", "time"))
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--engines", nargs="*", default=["quad", "pde", "mc"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--with-mc", action="store_true", help="also time the MC baseline in 'time' mode")
    args = parser.parse_args(argv)
    return tune(args) if args.mode == "tune" else time_runs(args)


if __name__ == "__main__":
    raise SystemExit(main())
