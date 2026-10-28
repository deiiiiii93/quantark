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
from .grid import GridConfig
from .kernels import CTMCKernel
from .market import load_fixture

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[2] / "docs/research/localvol_transition_quad"
FLOORS = {"pv": PV_BUDGET, "delta": 1e-3, "gamma": 1e-2, "vega": 1e-4, "theta": 1e-5, "rho": 1e-5, "carry": 1e-5}
PDE_LADDER = {"snowball_1y": [(800, 8), (1600, 16), (3200, 32), (6400, 64)],
              "snowball_2y": [(800, 8), (1600, 16), (3200, 32), (6400, 64)],
              "put_1y": [(800, 8), (1600, 16), (3200, 32)],
              "up_out_1y": [(800, 8), (1600, 16), (3200, 32)]}
QUAD_LADDER = [(400, 1), (400, 3), (400, 9), (400, 27)]
HALF_WIDTH_STD = 6.0


def budget(key: str, ref: float) -> float:
    """Spec accuracy budget per quantity (per unit notional)."""
    return PV_BUDGET if key == "pv" else max(0.01 * abs(ref), FLOORS[key])


def cases(market) -> dict:
    s0 = market.spot
    return {"snowball_1y": snowball_contract(s0, 1.0, 12), "snowball_2y": snowball_contract(s0, 2.0, 24),
            "put_1y": vanilla_contract(s0, 1.0, option="put"), "up_out_1y": up_out_contract(s0, 1.0, 12)}


PDE_SOURCES = ("adapters.py", "contract.py", "market.py")   # the production-PDE arm depends on these only


def source_hashes(names=None) -> dict:
    files = sorted(HERE.glob("*.py")) if names is None else [HERE / n for n in names]
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files}


def pde_params(points: int, steps_per_day: int) -> PDEParams:
    return PDEParams(accuracy="high", grid=PDEGrid(points=points, steps_per_day=steps_per_day,
                                                   max_points=20000, max_steps=200000))


def run_pde(contract, market, points, spd) -> dict:
    start = perf_counter()
    out = adapters.pde_full_risk(contract, market, pde_params(points, spd))
    return dict(values=out["values"], seconds=perf_counter() - start, timings=out["timings"])


def run_quad(contract, market, cells, subdivide) -> dict:
    start = perf_counter()
    engine = LocalVolQuad(CTMCKernel(), GridConfig(cells=cells, half_width_std=HALF_WIDTH_STD, subdivide=subdivide))
    out = engine.full_risk(contract, market)
    return dict(values=out["values"], seconds=perf_counter() - start, timings=out["timings"], cells=out["cells"])


def assess(case: dict) -> dict:
    """Reference = finest PDE; converged if the last PDE step and the QUAD cross-check are within 1/5 budget."""
    pde = case.get("pde", {})
    quad = case.get("quad", {})
    if len(pde) < 2:
        return {}
    keys = list(pde)
    finest, previous = pde[keys[-1]]["values"], pde[keys[-2]]["values"]
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
    engine = LocalVolQuad(CTMCKernel(), GridConfig(cells=400, half_width_std=HALF_WIDTH_STD, subdivide=3))
    for t in market.surface.time_grid[1:]:
        for m in moneyness:
            c = vanilla_contract(market.spot, float(t), strike_m=m, option="put")
            quad = engine.price(c, market)["values"]["pv"]
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
    hashes = {"pde": source_hashes(PDE_SOURCES), "quad": source_hashes()}
    data: dict = {"cases": {}}
    if args.resume and path.exists():
        data = json.loads(path.read_text())
        old = data.get("sources", {})
        for arm in ("pde", "quad"):
            if old.get(arm) not in (None, hashes[arm]):
                # results of an arm whose sources changed are stale: drop them, never mix
                for case in data["cases"].values():
                    case[arm] = {}
    data["sources"] = hashes
    data["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                           "machine": platform.machine(),
                           "fixture": "test/golden/modelvalidation/data/iv_surface_20240208.json"}
    market = load_fixture()
    data["market"] = {"spot": market.spot, "r": market.r, "q": market.q}
    if args.arm in ("dupire", "all"):
        data["dupire_gap"] = dupire_gap(market)
        path.write_text(json.dumps(data, indent=1))
        if args.arm == "dupire":
            return 0
    for name, contract in cases(market).items():
        if args.cases and name not in args.cases:
            continue
        case = data["cases"].setdefault(name, {"pde": {}, "quad": {}})
        if args.arm in ("pde", "all"):
            for points, spd in PDE_LADDER[name]:
                key = f"{points}x{spd}"
                if key not in case["pde"]:
                    case["pde"][key] = run_pde(contract, market, points, spd)
                    print(name, "pde", key, f"{case['pde'][key]['seconds']:.1f}s", flush=True)
                    path.write_text(json.dumps(data, indent=1))
        if args.arm in ("quad", "all"):
            for cells, sub in QUAD_LADDER:
                key = f"{cells}x{sub}"
                if key not in case["quad"]:
                    case["quad"][key] = run_quad(contract, market, cells, sub)
                    print(name, "quad", key, f"{case['quad'][key]['seconds']:.1f}s", flush=True)
                    path.write_text(json.dumps(data, indent=1))
        case["assessment"] = assess(case)
        path.write_text(json.dumps(data, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
