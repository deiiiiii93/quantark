#!/usr/bin/env python
"""Stage 11c -- Gate G5, the pre-flight grid-resolution sweep.

    .venv/bin/python example/mo_volmodels/11c_grid_preflight.py

`fdf3a70` promoted the spatial-grid resolution check from a warning to a
fail-closed ValidationError.  That is correct, and it converts a silent
accuracy loss into a MID-FLEET exception -- the residual risk being to
discover after ~143 CPU-hours that some inceptions were never priceable
(design spec section 9).

So: before the fleet launches, walk every (inception, remaining maturity)
operating point and BUILD THE GRID ONLY -- no solve.  The seam is the
solver's own declaration, so this tests what the fleet will actually build
rather than a parallel reimplementation:

    solver = PDEEngine._get_solver(product)
    tau     = solver._prepare_for_request(product, env)
    market  = solver.market_snapshot(product, env)
    request = solver.grid_request(product, market, tau)
    solver._grid_layer_binder(solver.params).bind(request, market)

`bind` is what calls `build_space`, and `build_space` is what raises.

SCOPE -- recorded in the artifact so it cannot overclaim.  This sweeps the
1-D PDE spatial grid under the flat-ATM-at-remaining-tenor environment.
That is not a subset chosen for convenience: stage 12 pins
`surface_engine_type=PDE` and `event_stats_engine_type=PDE` for EVERY
variant, so this grid is built on every replay day of every variant, and a
failure here grounds the whole fleet regardless of routing.  What it does
NOT cover is stated in `scope.not_covered`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

DEFAULT_HISTORY_DIR = PROJECT_ROOT / "example/mo_volmodels/data/history"
DEFAULT_INCEPTIONS = PROJECT_ROOT / "output/volmodel_backtest/inceptions.json"
DEFAULT_OUT = PROJECT_ROOT / "output/pde_convergence_gate/grid_preflight.json"

NOT_COVERED = [
    "2-D ADI variance-axis grid (n_v / variance_grid_mode) -- the vol-model "
    "solvers build it from calibrated Heston params, which this sweep does "
    "not load",
    "QUAD grid_points and unreachable-barrier filtering (flat_bsm_quad)",
    "full-grid and term-structure vol surfaces; the spatial grid's vol "
    "dependence enters through representative_vol, which this exercises at "
    "the day's ATM level",
]

_STAGE12 = None
_STAGE11 = None


def stage12():
    global _STAGE12
    if _STAGE12 is None:
        spec = importlib.util.spec_from_file_location(
            "mo_stage12_for_g5", HERE / "12_snowball_volmodel_backtest.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _STAGE12 = module
    return _STAGE12


def stage11():
    global _STAGE11
    if _STAGE11 is None:
        _STAGE11 = stage12().stage11()
    return _STAGE11


def build_artifact(
    *,
    points: List[Any],
    failures: List[Dict[str, Any]],
    scope: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """The G5 artifact.

    `n_operating_points` and `under_resolved` are the two keys the dashboard's
    `headline_g5` reads; it treats a missing or non-list `under_resolved` as
    INCOMPLETE rather than satisfied, so both are always written.
    """
    return {
        "schema_version": 1,
        "gate": "G5",
        "study": "snowball_volmodel_backtest",
        "n_operating_points": len(points),
        "under_resolved": list(failures),
        "scope": dict(scope),
        "config": dict(config),
    }


def _grid_builds(solver_engine, product, env) -> Optional[str]:
    """Build the grid for one operating point. None on success, else the error."""
    try:
        solver = solver_engine._get_solver(product)
        tau = solver._prepare_for_request(product, env)
        market = solver.market_snapshot(product, env)
        request = solver.grid_request(product, market, tau)
        # BasePDESolver exposes the engine-owned binder as a property;
        # GridLayerMixin adopters (the non-BasePDESolver vol engines) build
        # theirs on demand. Take whichever the solver actually owns rather
        # than constructing a third one, which would bind a config the fleet
        # never uses.
        binder = getattr(solver, "grid_binder", None)
        if binder is None:
            binder = solver._grid_layer_binder(getattr(solver, "params", None))
        binder.bind(request, market)
    except Exception as exc:  # noqa: BLE001 - the point is to catch and record
        return f"{type(exc).__name__}: {exc}"
    return None


def probe_one_inception(
    *,
    inception: str,
    history_dir: Path = DEFAULT_HISTORY_DIR,
    limit_days: Optional[int] = None,
    inceptions_path: Path = DEFAULT_INCEPTIONS,
    rate: float = 0.02,
) -> List[Dict[str, Any]]:
    """Sweep one inception's replay days. Returns the failing points."""
    s12 = stage12()
    s11 = stage11()

    records = json.loads(Path(inceptions_path).read_text())
    record = next(r for r in records if r["inception"] == inception)

    history = s12.surface_history(history_dir)
    spot = s12.load_spot_frame(history_dir)
    spot_by_date = {
        pd.Timestamp(row.date).date(): float(row.spot)
        for row in spot.itertuples(index=False)
    }
    calendar = s11.TradingCalendar.from_spot_csv(history_dir / "csi1000_spot.csv")
    start = date.fromisoformat(record["inception"])
    terms = s11.build_snowball_terms(start, calendar)

    product = s12.build_backtest_product(
        terms,
        initial_spot=float(record["initial_spot"]),
        coupon=float(record["coupon"]),
        notional=s12.NOTIONAL,
    )
    config = s12.make_engine_config(
        "flat_bsm",
        routing=s12.GateRouting("", None, {"flat_bsm": "pde"}, {}),
        calibration_cache_dir=None,
    )

    admitted = set(history.admitted_dates)
    window_end = min(terms.maturity_date, max(spot_by_date))
    days = [
        d for d in calendar.trading_days_between(start, window_end)
        if d in admitted and d in spot_by_date
    ]
    if limit_days is not None:
        days = days[: int(limit_days)]

    failures: List[Dict[str, Any]] = []
    for day in days:
        remaining = (terms.maturity_date - day).days / 365.0
        if remaining <= 0:
            continue
        env = s12.inception_pricing_env(
            history=history,
            inception=day,
            spot=spot_by_date[day],
            rate=rate,
            remaining_years=remaining,
        )
        engine = s12.create_pricing_engine(product, config)
        error = _grid_builds(engine, product, env)
        if error is not None:
            failures.append(
                {
                    "variant": "flat_bsm",
                    "inception": inception,
                    "date": day.isoformat(),
                    "tau_years": round(remaining, 6),
                    "spot": spot_by_date[day],
                    "error": error,
                }
            )
    return failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument("--inceptions", default=str(DEFAULT_INCEPTIONS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--rate", type=float, default=0.02)
    parser.add_argument(
        "--limit-days", type=int, default=None,
        help="cap replay days per inception (smoke only; a capped sweep is "
             "recorded as such and must not be read as a clean pre-flight)",
    )
    args = parser.parse_args(argv)

    history_dir = Path(args.history_dir)
    records = json.loads(Path(args.inceptions).read_text())
    all_points: List[str] = []
    all_failures: List[Dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        inception = record["inception"]
        failures = probe_one_inception(
            inception=inception,
            history_dir=history_dir,
            limit_days=args.limit_days,
            inceptions_path=Path(args.inceptions),
            rate=float(args.rate),
        )
        # Re-derive the day count the sweep actually visited.
        s12 = stage12()
        s11 = stage11()
        calendar = s11.TradingCalendar.from_spot_csv(
            history_dir / "csi1000_spot.csv"
        )
        history = s12.surface_history(history_dir)
        spot = s12.load_spot_frame(history_dir)
        spot_by_date = {
            pd.Timestamp(row.date).date(): float(row.spot)
            for row in spot.itertuples(index=False)
        }
        terms = s11.build_snowball_terms(date.fromisoformat(inception), calendar)
        admitted = set(history.admitted_dates)
        window_end = min(terms.maturity_date, max(spot_by_date))
        days = [
            d for d in calendar.trading_days_between(
                date.fromisoformat(inception), window_end
            )
            if d in admitted and d in spot_by_date
            and (terms.maturity_date - d).days > 0
        ]
        if args.limit_days is not None:
            days = days[: int(args.limit_days)]
        all_points.extend(f"{inception}:{d.isoformat()}" for d in days)
        all_failures.extend(failures)
        print(
            f"  [{index}/{len(records)}] {inception}: {len(days)} points, "
            f"{len(failures)} under-resolved",
            flush=True,
        )

    doc = build_artifact(
        points=all_points,
        failures=all_failures,
        scope={
            "covered": "1-D PDE spatial grid on every admitted replay day of "
                       "every inception, under the flat-ATM-at-remaining-tenor "
                       "environment",
            "why_this_grid": "stage 12 pins surface_engine_type=PDE and "
                             "event_stats_engine_type=PDE for EVERY variant, so "
                             "this grid is built on every replay day regardless "
                             "of routing",
            "not_covered": NOT_COVERED,
            "capped": args.limit_days is not None,
        },
        config={
            "history_dir": str(history_dir),
            "inceptions": str(args.inceptions),
            "rate": float(args.rate),
            "limit_days": args.limit_days,
        },
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"[G5] {doc['n_operating_points']} operating points, "
          f"{len(doc['under_resolved'])} under-resolved")
    print(f"[G5] wrote {out}")
    return 0 if not doc["under_resolved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
