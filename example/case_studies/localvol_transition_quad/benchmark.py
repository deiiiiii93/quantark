"""Matched-accuracy tuning (spec D13, T7) and the cold/warm full-risk benchmark (spec D14, T8).

    python -m example.case_studies.localvol_transition_quad.benchmark tune [--cases ...] [--engines ...]
    python -m example.case_studies.localvol_transition_quad.benchmark time [--repeats 3]

``tune`` runs a fixed configuration ladder per engine and case against the D12
reference and keeps the cheapest configuration whose WORST output is within
budget (the binding quantity is recorded); ``--resume`` keeps only rows whose
fingerprint (terms, config, sources, reference) still matches. ``time`` re-times
the tuned configurations in ONE window: an untimed warm-up on tuned configs and an
untimed validation-mode pass of the QUAD config, then per repeat a COLD workload
(fresh engine objects, empty operator caches) for every arm with the order
alternated. Arms: the tuned QUAD and PDE, QUAD at the first rung within half the
budget, and (Snowballs past the last pillar) QUAD with the flat-slab rate cache
off. ``warm_moved`` re-runs QUAD with retained kernels and the PDE as shipped at a
distinct re-anchored spot per repeat (s0 x 1.005/1.006/1.007), so no cache can
hit. Identical-input replay is not timed. Run with OPENBLAS/OMP/VECLIB_*_NUM_THREADS=1.
"""
from __future__ import annotations

import argparse
import hashlib
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

from . import adapters, kernels
from .engine import OUTPUTS, LocalVolQuad
from .grid import GridConfig, UnresolvedGrid
from .kernels import CTMCKernel
from .market import load_fixture
from .reference import OUT_DIR, budget, cases, source_hashes

QUAD_LADDER = [(c, hw) for hw in (6.0, 8.0) for c in (200, 250, 300, 350, 400, 500, 600)]
PDE_LADDER = [(100, 1), (150, 1), (200, 1), (150, 2), (200, 2), (300, 2), (200, 4), (300, 4), (400, 4)]
# Gamma SE alone (not bias) at 2^20 paths is 13-20x its budget; 2 SE <= budget would need ~2^29.4 (1y) /
# ~2^30.7 (2y) paths, far beyond D13's 2^22 cap: the ladder stops at 2^20.
MC_LADDER = [(1, 2 ** 16), (1, 2 ** 18), (1, 2 ** 20), (4, 2 ** 16)]
MC_BATCH = 2 ** 14
MOVES = (0.005, 0.006, 0.007)     # warm_moved: a DISTINCT moved spot per repeat, so no cache can hit
MARGIN = 0.5                      # in-window QUAD row at the first rung within half the budget
PDE_SOURCES = ("adapters.py", "contract.py", "market.py")
QUAD_SOURCES = ("contract.py", "market.py", "grid.py", "kernels.py", "engine.py")


def row_fingerprint(engine: str, contract, config, reference_case: dict) -> str:
    """Contract terms, engine config, the sources the numbers depend on and the reference they are scored
    against. A resumed row is kept only when this matches (missing = stale)."""
    names = QUAD_SOURCES if engine == "quad" else PDE_SOURCES
    text = json.dumps([engine, repr(contract), list(config), source_hashes(names),
                       reference_case["assessment"]["reference"]], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


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


def timed_arm(engine: str, config, contract, market) -> dict:
    """A cold full risk run; ``quad_no_slab_cache`` disables the flat-slab rate cache (D14 slab ablation)."""
    if engine != "quad_no_slab_cache":
        return full_risk(engine, config, contract, market)
    original = kernels.SurfaceSampler.clamped_side
    kernels.SurfaceSampler.clamped_side = lambda self, t: None
    try:
        return full_risk("quad", config, contract, market)
    finally:
        kernels.SurfaceSampler.clamped_side = original


def score(values: dict, ref: dict, stderr: dict | None = None) -> dict:
    """Worst error as a fraction of budget; MC adds 2 SE (spec D13)."""
    ratio = {k: (abs(values[k] - ref[k]) + (2 * stderr[k] if stderr else 0.0)) / budget(k, ref[k]) for k in OUTPUTS}
    binding = max(ratio, key=ratio.get)
    return dict(ratio=ratio, worst=ratio[binding], binding=binding, meets=ratio[binding] <= 1.0)


def ladder(engine: str):
    return {"quad": QUAD_LADDER, "pde": PDE_LADDER, "mc": MC_LADDER}[engine]


def tune(args) -> int:
    path = OUT_DIR / "tuning.json"
    refs = json.loads((OUT_DIR / "reference.json").read_text())["cases"]
    market = load_fixture()
    all_cases = cases(market)
    data = json.loads(path.read_text()) if args.resume and path.exists() else {"cases": {}}
    for name, case in data.get("cases", {}).items():          # drop stale rows, never mix provenance
        for engine in ("quad", "pde", "mc"):
            if engine in case and name in all_cases:
                case[engine] = {k: r for k, r in case[engine].items()
                                if r.get("fingerprint") == row_fingerprint(engine, all_cases[name],
                                                                           r["config"], refs[name])}
    data["sources"] = {"quad": source_hashes(QUAD_SOURCES), "pde_mc": source_hashes(PDE_SOURCES)}
    for name, contract in all_cases.items():
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
                rows[key]["fingerprint"] = row_fingerprint(engine, contract, config, refs[name])
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
    # untimed process warm-up on TUNED configurations: imports, BLAS/LAPACK first calls, surface loading
    warm = all_cases["put_1y"]
    quad_engine(*tuning["put_1y"]["quad_chosen"]).price(warm, market)
    adapters.pde_full_risk(warm, market, pde_params(*tuning["put_1y"]["pde_chosen"]))
    for name in args.cases or list(all_cases):
        contract = all_cases[name]
        ref = refs[name]["assessment"]["reference"]
        chosen = {e: tuning[name].get(f"{e}_chosen") for e in ("quad", "pde", "mc")}
        arms = {e: (e, tuple(chosen[e])) for e in ("quad", "pde", "mc") if chosen[e] is not None}
        if "mc" in arms and not args.with_mc:
            del arms["mc"]
        # in-window companions for every ratio the report publishes
        margin = sorted((r for r in tuning[name]["quad"].values() if r.get("meets") and r["worst"] <= MARGIN),
                        key=lambda r: r["seconds"])
        if margin:
            arms["quad_margin"] = ("quad", tuple(margin[0]["config"]))
        if contract.kind == "snowball" and contract.maturity > float(market.surface.time_grid[-1]):
            arms["quad_no_slab_cache"] = ("quad_no_slab_cache", tuple(chosen["quad"]))
        if "quad" in arms:                       # D16: the timed configuration once in validation mode, untimed
            kernel = CTMCKernel()
            out = LocalVolQuad(kernel, GridConfig(cells=chosen["quad"][0], half_width_std=chosen["quad"][1]),
                               validate=True).price(contract, market)
            validation = dict(engine=out["diagnostics"], kernel_max={
                k: max(d[k] for d in kernel.diagnostics) for k in ("mass", "forward", "semigroup")})
        rows = {a: dict(config=list(cfg), cold=[], valuations=[]) for a, (_, cfg) in arms.items()}
        names = list(arms)
        for rep in range(args.repeats):
            for a in (names if rep % 2 == 0 else names[::-1]):
                engine, cfg = arms[a]
                start = perf_counter()
                out = timed_arm(engine, cfg, contract, market)
                rows[a]["cold"].append(perf_counter() - start)
                rows[a]["valuations"].append(out["timings"])
                rows[a]["values"] = out["values"]
                rows[a]["peak_rss_after"] = _peak_rss()
        # warm_moved: retained QUAD kernel / PDE as shipped, a distinct re-anchored spot per repeat
        for a in ("quad", "pde"):
            if a not in arms:
                continue
            engine_obj = quad_engine(*chosen["quad"]) if a == "quad" else None
            if engine_obj is not None:
                engine_obj.full_risk(contract, market)
            times = []
            for move in MOVES[:args.repeats]:
                moved_market = replace(market, spot=market.spot * (1 + move))
                moved = replace(contract, s0=market.spot * (1 + move))
                start = perf_counter()
                if engine_obj is not None:
                    engine_obj.full_risk(moved, moved_market)
                else:
                    adapters.pde_full_risk(moved, moved_market, pde_params(*chosen["pde"]))
                times.append(perf_counter() - start)
            rows[a]["warm_moved"] = times
            rows[a]["warm_moves"] = list(MOVES[:args.repeats])
        for a, row in rows.items():
            row["median_cold"] = statistics.median(row["cold"])
            row["accuracy"] = score(row["values"], ref)
        if "quad" in rows:
            rows["quad"]["validation"] = validation
        for a in ("pde", "mc"):
            if "quad" in rows and a in rows:
                rows[f"speedup_vs_{a}"] = rows[a]["median_cold"] / rows["quad"]["median_cold"]
        if "quad_margin" in rows and "pde" in rows:
            rows["speedup_margin_vs_pde"] = rows["pde"]["median_cold"] / rows["quad_margin"]["median_cold"]
        if "quad_no_slab_cache" in rows and "pde" in rows:
            rows["speedup_no_slab_cache_vs_pde"] = (rows["pde"]["median_cold"]
                                                    / rows["quad_no_slab_cache"]["median_cold"])
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
