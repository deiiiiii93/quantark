"""High-resolution references for every benchmarked case (spec D12).

Runs the production LocalVolSnowballPDESolver / LocalVolPDESolver / LocalVolBarrierPDESolver
full risk on a refinement ladder with explicitly raised grid caps, and the CTMC QUAD on a
nested x3 ladder. Writes JSON incrementally; ``--resume`` skips rungs already present only
when the recorded source hashes match.

    python -m example.case_studies.localvol_transition_quad.reference [--arm pde|quad|all] [--resume]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import ndtr

from quantark.asset.equity.engine.pde.grid.config import GridConfig as PDEGrid
from quantark.asset.equity.param import PDEParams

from . import adapters
from .contract import snowball_contract, up_out_contract, vanilla_contract
from .engine import OUTPUTS, PV_BUDGET, LocalVolQuad
from .grid import GridConfig, UnresolvedGrid
from .kernels import CTMCKernel
from .market import FIXTURE, load_fixture

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[2] / "docs/research/localvol_transition_quad"
FLOORS = {"pv": PV_BUDGET, "delta": 1e-3, "gamma": 1e-2, "vega": 1e-4, "theta": 1e-5, "rho": 1e-5, "carry": 1e-5}
PDE_LADDER = {"snowball_1y": [(800, 8), (1600, 16), (3200, 32), (6400, 64)],
              "snowball_2y": [(800, 8), (1600, 16), (3200, 32), (6400, 64)],
              "put_1y": [(800, 8), (1600, 16), (3200, 32)],
              "up_out_1y": [(800, 8), (1600, 16), (3200, 32)]}
QUAD_LADDER = [(400, 1), (400, 3), (400, 9), (400, 27)]
HALF_WIDTH_STD = 8.0     # reference domain: the ATM-scaled width must also cover the steep short-dated wings


def budget(key: str, ref: float) -> float:
    """Spec accuracy budget per quantity (per unit notional)."""
    return PV_BUDGET if key == "pv" else max(0.01 * abs(ref), FLOORS[key])


def cases(market) -> dict:
    s0 = market.spot
    return {"snowball_1y": snowball_contract(s0, 1.0, 12), "snowball_2y": snowball_contract(s0, 2.0, 24),
            "put_1y": vanilla_contract(s0, 1.0, option="put"), "up_out_1y": up_out_contract(s0, 1.0, 12)}


PDE_SOURCES = ("adapters.py", "contract.py", "market.py")   # the production-PDE arm depends on these only
QUAD_SOURCES = ("contract.py", "market.py", "grid.py", "kernels.py", "engine.py")   # and the QUAD arm on these


def source_hashes(names=None) -> dict:
    files = sorted(HERE.glob("*.py")) if names is None else [HERE / n for n in names]
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files}


def pde_params(points: int, steps_per_day: int) -> PDEParams:
    return PDEParams(accuracy="high", grid=PDEGrid(points=points, steps_per_day=steps_per_day,
                                                   max_points=20000, max_steps=200000))


def quad_config(cells: int, subdivide: int) -> GridConfig:
    return GridConfig(cells=cells, half_width_std=HALF_WIDTH_STD, subdivide=subdivide)


def fingerprint(arm: str, contract, config) -> str:
    """Everything a row's numbers depend on: contract terms, engine parameters, fixture bytes, sources.

    A resumed row is kept only when its fingerprint matches; a missing one is stale.
    """
    sources = source_hashes(PDE_SOURCES if arm == "pde" else QUAD_SOURCES)
    fixture = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()[:16]
    params = repr(pde_params(*config)) if arm == "pde" else repr(quad_config(*config))
    text = json.dumps([arm, repr(contract), params, fixture, sources], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def run_pde(contract, market, points, spd) -> dict:
    start = perf_counter()
    out = adapters.pde_full_risk(contract, market, pde_params(points, spd))
    return dict(values=out["values"], seconds=perf_counter() - start, timings=out["timings"])


def run_quad(contract, market, cells, subdivide) -> dict:
    start = perf_counter()
    try:
        out = LocalVolQuad(CTMCKernel(), quad_config(cells, subdivide)).full_risk(contract, market)
    except UnresolvedGrid as exc:                # recorded, never fatal (spec D16)
        return dict(status="refused", reason=str(exc))
    return dict(values=out["values"], seconds=perf_counter() - start, timings=out["timings"], cells=out["cells"])


def _ordered(rows: dict, ladder) -> list[dict]:
    """Priced rungs in LADDER order (coarse to fine), whatever order they were computed in."""
    keys = [f"{a}x{b}" for a, b in ladder]
    return [rows[k] for k in keys if k in rows and "values" in rows[k]]


def assess(case: dict, name: str | None = None) -> dict:
    """Reference = finest PDE; converged if the last PDE step and the QUAD cross-check are within 1/5 budget."""
    pde_ladder = PDE_LADDER.get(name) if name else None
    pde = _ordered(case.get("pde", {}), pde_ladder) if pde_ladder else list(case.get("pde", {}).values())
    quad_rows = _ordered(case.get("quad", {}), QUAD_LADDER) if name else list(case.get("quad", {}).values())
    quad = {i: row for i, row in enumerate(quad_rows)}
    if len(pde) < 2:
        return {}
    finest, previous = pde[-1]["values"], pde[-2]["values"]
    verdict = {"reference": finest, "pde_step": {}, "quad_gap": {}, "budget": {}}
    for k in OUTPUTS:
        b = budget(k, finest[k])
        verdict["budget"][k] = b
        verdict["pde_step"][k] = finest[k] - previous[k]
        if quad:
            verdict["quad_gap"][k] = quad[list(quad)[-1]]["values"][k] - finest[k]
    verdict["pde_converged"] = all(abs(verdict["pde_step"][k]) <= verdict["budget"][k] / 5 for k in OUTPUTS)
    verdict["quad_confirms"] = bool(quad) and all(abs(verdict["quad_gap"][k]) <= verdict["budget"][k] / 5
                                                  for k in OUTPUTS)
    return verdict


def black_scholes_put(spot, strike, t, r, q, vol) -> float:
    d1 = (math.log(spot / strike) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    return strike * math.exp(-r * t) * ndtr(-d2) - spot * math.exp(-q * t) * ndtr(-d1)


def dupire_gap(market, moneyness=(0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)) -> list[dict]:
    """QUAD European puts on the Dupire surface versus BS(input iv, flat r, flat q), per unit notional.

    Reported, not gated (spec AC4): a 6 x 33 finite-difference Dupire surface with
    bilinear interpolation and flat extrapolation does not reprice its input exactly
    even in the continuum limit, so the gap has a non-zero floor.
    """
    rows = []
    # short maturities on a steep-winged surface need a wider domain than the ATM-scaled default
    engine = LocalVolQuad(CTMCKernel(), GridConfig(cells=600, half_width_std=10.0, subdivide=3))
    for t in market.surface.time_grid[1:]:
        for m in moneyness:
            c = vanilla_contract(market.spot, float(t), strike_m=m, option="put")
            try:
                quad = engine.price(c, market)["values"]["pv"]
            except UnresolvedGrid as exc:            # the harness records refusals (spec D16)
                rows.append(dict(maturity=float(t), moneyness=m, refused=str(exc)))
                continue
            iv = float(market.iv_surface.get_vol(c.strike, float(t), market.spot))
            bs = black_scholes_put(market.spot, c.strike, float(t), market.r, market.q, iv) / market.spot
            rows.append(dict(maturity=float(t), moneyness=m, implied_vol=iv, quad=quad, black_scholes=bs,
                             gap=quad - bs, gap_vol_points=(quad - bs) / max(vega_per_vol(market, c, iv), 1e-300)))
    return rows


def vega_per_vol(market, c, iv) -> float:
    """BS vega per unit vol, per unit notional (to express a price gap in vol points)."""
    t = c.maturity
    d1 = (math.log(market.spot / c.strike) + (market.r - market.q + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    return math.exp(-market.q * t) * math.sqrt(t) * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("pde", "quad", "dupire", "all"), default="all")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default=str(OUT_DIR / "reference.json"))
    args = parser.parse_args(argv)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    market = load_fixture()
    all_cases = cases(market)
    data: dict = {"cases": {}}
    if args.resume and path.exists():
        data = json.loads(path.read_text())
        # keep only rows whose fingerprint still matches (missing fingerprint = stale); never mix
        for name, case in data.get("cases", {}).items():
            for arm, ladder in (("pde", PDE_LADDER.get(name, [])), ("quad", QUAD_LADDER)):
                rows = case.get(arm, {})
                case[arm] = {f"{a}x{b}": rows[f"{a}x{b}"] for a, b in ladder
                             if name in all_cases and f"{a}x{b}" in rows
                             and rows[f"{a}x{b}"].get("fingerprint") == fingerprint(arm, all_cases[name], (a, b))}
        dupire_key = fingerprint("quad", "dupire", (600, 3))
        if data.get("dupire_gap_fingerprint") != dupire_key:
            data.pop("dupire_gap", None)
    data["sources"] = {"pde": source_hashes(PDE_SOURCES), "quad": source_hashes()}
    data["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                           "machine": platform.machine(),
                           "fixture": "test/golden/modelvalidation/data/iv_surface_20240208.json"}
    data["market"] = {"spot": market.spot, "r": market.r, "q": market.q}
    if args.arm in ("dupire", "all"):
        data["dupire_gap"] = dupire_gap(market)
        data["dupire_gap_fingerprint"] = fingerprint("quad", "dupire", (600, 3))
        path.write_text(json.dumps(data, indent=1))
    for name, contract in all_cases.items():
        if args.arm == "dupire" or (args.cases and name not in args.cases):
            continue
        case = data["cases"].setdefault(name, {"pde": {}, "quad": {}})
        for arm, ladder, runner in (("pde", PDE_LADDER[name], run_pde), ("quad", QUAD_LADDER, run_quad)):
            if args.arm not in (arm, "all"):
                continue
            for config in ladder:
                key = f"{config[0]}x{config[1]}"
                if key not in case[arm]:
                    row = runner(contract, market, *config)
                    row["fingerprint"] = fingerprint(arm, contract, config)
                    case[arm][key] = row
                    print(name, arm, key, row.get("status", "priced"), f"{row.get('seconds', 0):.1f}s", flush=True)
                    path.write_text(json.dumps(data, indent=1))
    for name, case in data["cases"].items():          # every case, so no stale assessment survives
        case["assessment"] = assess(case, name)
    path.write_text(json.dumps(data, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
