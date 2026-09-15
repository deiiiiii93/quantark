"""Stage 02: the cells over the simulated batches.

    .venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py --quick
    nohup caffeinate -i -m -s .venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py \\
        --workers 4 --batch-paths 250 --disk-cache > output/snowball_simulated_paths/fleet.log 2>&1 &

Cells are {baseline flat q, term_flat_q, term_opt_tail} x {front, far}.  Each
cell runs the bootstrap batch and the stress set on ``--provider`` (default
``per_date``: exact repricing on the PDE engine, one solve from maturity back
to each state's date with that date's real dividend object), then the first
``--exact-paths`` bootstrap paths on exact QUAD repricing -- the engine check,
paired with the cell on the same paths -- and optionally ``--check-paths``
on the QUAD spot ladder.  Every bootstrap run is oracle-checked on
``--oracle-paths`` single paths against the replay engine, every check run
on the first of them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _sim_common as C  # noqa: E402


def _stage(name: str):
    """Load a sibling stage module (their names start with a digit, so no plain import)."""
    spec = importlib.util.spec_from_file_location(
        f"snowball_simulated_paths_{name}", Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


S01 = _stage("01_build_paths")

from quantark.backtest.futures_ledger import FuturesRollPolicy  # noqa: E402
from quantark.backtest.replay import HedgeSpec, ReplayProduct  # noqa: E402
from quantark.backtest.replay.engine_factory import create_pricing_engine  # noqa: E402
from quantark.backtest.simulation import (  # noqa: E402
    CacheConfig, EnsembleConfig, GateConfig, MarketPath, PricingProviderConfig, day_chain,
    dividend_yield_for_day, run_ensemble, run_oracle,
)
from quantark.backtest.simulation.hedge import day_active_contract  # noqa: E402
from quantark.backtest.simulation.pricing.base import GateFailure  # noqa: E402
from quantark.backtest.simulation.results import EnsembleResults  # noqa: E402
from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy  # noqa: E402
from quantark.backtest.transaction_costs import ProportionalCostModel, ZeroCostModel  # noqa: E402
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402

PROVIDERS = ("per_date", "exact", "life_surface", "ladder")
#: Providers priced by the PDE engine; the others price on QUAD.
PDE_PROVIDERS = ("per_date", "life_surface")


def study_terms(calendar: pd.DatetimeIndex, *, maturity_months: int, lockout_months: int):
    """The q study's term sheet on the simulated calendar, inception on its first day."""
    days = [pd.Timestamp(d).date() for d in calendar]
    return C.Q.build_terms(days[0], C.Q.TradingCalendar(days), maturity_months=maturity_months,
                           lockout_months=lockout_months)


def start_env(paths: MarketPath, model: str, *, quad_grid: int, multiplier: float) -> Tuple[PricingEnvironment, str]:
    """Path 0's day-0 pricing environment under ``model`` (the fair coupon's market)."""
    chain = day_chain(paths, 0, multiplier=multiplier)
    code, _ = day_active_contract(chain, FuturesRollPolicy(roll_days_before_expiry=C.Q.ROLL_DAYS_BEFORE_EXPIRY), None)
    spot, rate = float(paths.spot[0, 0]), float(paths.rate[0, 0])
    dividend = dividend_yield_for_day(
        chain, 0, spot=spot, rate=rate, engine_config=C.engine_config(model, "quad", quad_grid=quad_grid),
        active_contract=code, curve_tenors=paths.tenor_grid, curve_carry=paths.carry[0, 0],
    )
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name=C.Q.UNDERLYING_NAME),
        vol_surface=FlatVolSurface(volatility=float(paths.atm_vol[0, 0])), rate_curve=FlatRateCurve(rate=rate),
        div_yield=dividend, valuation_date=pd.Timestamp(paths.dates[0]).to_pydatetime(),
    )
    return env, code


def fair_coupon(paths: MarketPath, terms, *, model: str, quad_grid: int):
    """The coupon that prices the product to zero at the start state under ``model`` (QUAD)."""
    env, _ = start_env(paths, model, quad_grid=quad_grid, multiplier=C.Q.FUTURES_MULTIPLIER)
    config = C.engine_config(model, "quad", quad_grid=quad_grid)
    s0 = float(paths.spot[0, 0])

    def pv_at(coupon: float) -> float:
        product = C.Q.build_product(terms, s0, coupon)
        return float(create_pricing_engine(product, config).price(product, env))

    return C.Q.solve_fair_coupon(pv_at, notional=terms.notional)


def _pricing(provider: str, *, disk_dir: Optional[str], gate_override: Optional[Dict[str, Any]]) -> PricingProviderConfig:
    cache = CacheConfig(memory_bytes=C.STATE_CACHE_BYTES, disk_dir=disk_dir)
    if provider == "life_surface":
        return PricingProviderConfig(
            provider="life_surface", cache=cache, gate=GateConfig(**(gate_override or C.GATE_SURFACE)),
            vol_step=C.VOL_STEP, q_step=C.Q_STEP, surface_cache_bytes=C.SURFACE_CACHE_BYTES,
        )
    if provider == "ladder":
        return PricingProviderConfig(
            provider="repricing", cache=cache, gate=GateConfig(**(gate_override or C.GATE_LADDER)),
            spot_step=C.SPOT_STEP, vol_step=C.VOL_STEP, q_step=C.Q_STEP,
        )
    if provider in ("per_date", "exact"):
        return PricingProviderConfig(provider="repricing", cache=cache,
                                     gate=GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0))
    raise ValidationError(f"provider must be one of {PROVIDERS}, got {provider!r}")


def _coupon_of(product) -> float:
    """The KO rate ``create_standard_snowball`` stored: ``BarrierConfig.ko_rate`` (a float, or one per observation)."""
    rate = product.barrier_config.ko_rate
    return float(rate[0] if isinstance(rate, (list, tuple)) else rate)


def cell_config(
    product, model: str, hedge: str, *, provider: str, cost_bp: float, workers: int, batch_paths: Optional[int],
    quad_grid: int, disk_dir: Optional[str] = None, gate_override: Optional[Dict[str, Any]] = None,
    spot_range: Optional[Tuple[float, float]] = None,
) -> EnsembleConfig:
    """One cell: the q study's product, model and hedge policy on the named provider.

    ``spot_range`` is the life surface's domain as fractions of the initial
    spot (default ``C.SURFACE_SPOT_RANGE``); it is recorded in the metadata
    and the config fingerprint.
    """
    if hedge not in C.Q.HEDGE_POLICIES:
        raise ValidationError(f"unknown hedge {hedge!r}; one of {tuple(C.Q.HEDGE_POLICIES)}")
    engine = "pde" if provider in PDE_PROVIDERS else "quad"
    spot_range = tuple(float(v) for v in (spot_range if spot_range is not None else C.SURFACE_SPOT_RANGE))
    return EnsembleConfig(
        products=[ReplayProduct(product=product, quantity=C.Q.PRODUCT_QUANTITY, position_id=1,
                                has_lifecycle=True, initial_price=0.0)],
        engine_config=C.engine_config(model, engine, quad_grid=quad_grid, s0=float(product.initial_price),
                                      spot_range=spot_range),
        hedge=HedgeSpec(kind="futures", multiplier=C.Q.FUTURES_MULTIPLIER, roll_policy=C.Q.HEDGE_POLICIES[hedge]()),
        strategy=AutocallableDeltaHedgeStrategy(delta_threshold=0.0, hedge_ratio=1.0, target_delta=0.0),
        transaction_cost_model=ProportionalCostModel(commission_rate=float(cost_bp) * 1e-4) if cost_bp else ZeroCostModel(),
        pricing=_pricing(provider, disk_dir=disk_dir, gate_override=gate_override),
        underlying=C.Q.UNDERLYING_NAME, workers=int(workers), batch_paths=batch_paths,
        metadata={"study": "snowball_simulated_paths", "model": model, "hedge": hedge, "provider": provider,
                  "engine": engine, "cost_bp": float(cost_bp), "quad_grid": int(quad_grid),
                  "coupon": _coupon_of(product), "spot_range": list(spot_range)},
    )


def config_fingerprint(config: EnsembleConfig, paths: MarketPath) -> str:
    """blake2b over the stated settings and the batch; never Python's hash."""
    p = config.pricing
    parts = (
        config.metadata.get("model"), config.metadata.get("hedge"), config.metadata.get("provider"),
        config.metadata.get("cost_bp"), config.metadata.get("quad_grid"), config.metadata.get("coupon"),
        tuple(config.metadata.get("spot_range") or ()), p.mode, p.spot_step, p.vol_step, p.q_step, p.surface_cache_bytes,
        p.gate.sample_states, p.gate.pv_tolerance_bp, p.gate.delta_tolerance_hands,
        repr(config.engine_config), paths.fingerprint(), paths.n_paths,
    )
    return hashlib.blake2b(repr(parts).encode(), digest_size=16).hexdigest()


def oracle_tolerances(config: EnsembleConfig) -> Dict[str, float]:
    """The gate's budget in the oracle's units: currency, delta units, hands."""
    gate = config.pricing.gate
    notional = sum(abs(float(bp.quantity)) * float(bp.product.initial_price) * float(bp.product.contract_multiplier)
                   for bp in config.products)
    if config.pricing.mode == "exact":
        return {"pv_tolerance": 0.0, "delta_tolerance": 0.0, "contracts_tolerance": 0.0}
    return {
        "pv_tolerance": float(gate.pv_tolerance_bp) * 1e-4 * notional,
        "delta_tolerance": float(gate.delta_tolerance_hands) * float(config.hedge.multiplier),
        "contracts_tolerance": float(gate.delta_tolerance_hands),
    }


def batch_for(n_paths: int, workers: int, batch_paths: Optional[int]) -> Optional[int]:
    """This run's batch size: never more than one batch per worker needs.

    One ``--batch-paths`` for 2,000 bootstrap paths would leave a 40-path
    check run as a single batch on one worker.  Batching is bit-inert and
    not part of the resume fingerprint, so sizing it per run changes no
    number.
    """
    if batch_paths is None:
        return None
    return max(1, min(int(batch_paths), math.ceil(int(n_paths) / max(1, int(workers)))))


def _config_record(config: EnsembleConfig, paths: MarketPath, fingerprint: str) -> Dict[str, Any]:
    return {
        "fingerprint": fingerprint, "metadata": config.metadata, "mode": config.pricing.mode,
        "gate": {"sample_states": config.pricing.gate.sample_states, "pv_tolerance_bp": config.pricing.gate.pv_tolerance_bp,
                 "delta_tolerance_hands": config.pricing.gate.delta_tolerance_hands},
        "steps": {"spot_step": config.pricing.spot_step, "vol_step": config.pricing.vol_step, "q_step": config.pricing.q_step},
        "workers": config.workers, "batch_paths": config.batch_paths, "n_paths": paths.n_paths,
        "path_fingerprint": paths.fingerprint(),
    }


def run_cell(paths: MarketPath, config: EnsembleConfig, out_dir, *, resume: bool,
             oracle_paths: Sequence[int]) -> Dict[str, Any]:
    """Run one cell over ``paths``, persist it, spot-check single paths against the replay engine.

    A run that misses its gate produces no results (the provider's claim
    failed); it is recorded in ``run.json`` with ``failed`` set and the
    fleet carries on with the next run.  With ``resume``, a run whose
    ``config.json`` fingerprint matches is skipped when its ``run.json``
    exists (a failed one included), and reuses its persisted results and
    runs only the oracle when the results exist but ``run.json`` does not
    (a run interrupted in its oracle).
    """
    out_dir = Path(out_dir)
    fingerprint = config_fingerprint(config, paths)
    config_path, run_path = out_dir / "config.json", out_dir / "run.json"
    matches = resume and config_path.exists() and C.read_json(config_path).get("fingerprint") == fingerprint
    if matches and run_path.exists():
        return {**C.read_json(run_path), "skipped": True}
    started = time.perf_counter()
    cell = config.metadata.get("model") + "__" + config.metadata.get("hedge")
    resumed = bool(matches and (out_dir / "manifest.json").exists())
    if resumed:
        results = EnsembleResults.from_dir(out_dir)
    else:
        try:
            results = run_ensemble(config, paths)
        except GateFailure as failure:
            run = {
                "cell": cell, "provider": config.metadata.get("provider"), "n_paths": paths.n_paths,
                "seconds": time.perf_counter() - started, "gate": failure.report.as_dict(), "oracle": [],
                "oracle_tolerances": oracle_tolerances(config), "skipped": False, "failed": True,
                "resumed_results": False,
            }
            C.write_json(config_path, _config_record(config, paths, fingerprint))
            C.write_json(run_path, run)
            return run
        results.to_dir(out_dir)
        C.write_json(config_path, _config_record(config, paths, fingerprint))
    # The book's mark on day 0.  Cells start from the traded price (0), and
    # the coupon is solved on QUAD, so under the PDE this is the engine gap
    # at inception; it is recorded rather than buried in terminal P&L.
    day0_mtm = float(results.cube.product_mtm[0, 0])
    single = replace(config, workers=1, batch_paths=None)
    tolerances = oracle_tolerances(config)
    reports = []
    for i in oracle_paths:
        try:
            report = run_oracle(single, paths.take([int(i)]), 0, **tolerances)
        except GateFailure as failure:          # the one-path re-run has its own gate and reservoir
            reports.append({"path": int(i), "passed": False, "gate": failure.report.as_dict()})
            continue
        reports.append({"path": int(i), **report.as_dict()})
    run = {
        "cell": cell, "provider": config.metadata.get("provider"),
        "n_paths": paths.n_paths, "seconds": time.perf_counter() - started, "gate": results.manifest["gate"],
        "engine_calls": results.manifest["engine_calls"], "solves": results.manifest.get("solves", 0),
        "oracle": reports, "oracle_tolerances": tolerances, "skipped": False, "failed": False,
        "resumed_results": resumed,
        "day0_product_mtm": day0_mtm, "day0_book_mark_bp": day0_mtm / results.notional * 1e4,
    }
    C.write_json(run_path, run)
    return run


def parse_cells(values: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    if not values:
        return [(m, h) for m in C.MODELS for h in C.HEDGES]
    out = []
    for value in values:
        model, _, hedge = value.partition(":")
        if model not in C.MODELS or hedge not in C.HEDGES:
            raise ValidationError(f"cell {value!r} must be model:hedge with model in {C.MODELS} and hedge in {C.HEDGES}")
        out.append((model, hedge))
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=C.DEFAULT_OUT_DIR)
    parser.add_argument("--cells", nargs="+", default=None, help="model:hedge, default = the full grid")
    parser.add_argument("--provider", choices=PROVIDERS, default="per_date",
                        help="the bootstrap and stress runs' provider; per_date = exact PDE, exact = exact QUAD")
    parser.add_argument("--check-paths", type=int, default=C.CHECK_PATHS, help="QUAD spot-ladder subset (0 = none)")
    parser.add_argument("--exact-paths", type=int, default=C.EXACT_CHECK_PATHS, help="exact QUAD check subset (0 = none)")
    parser.add_argument("--spot-range", type=float, nargs=2, default=list(C.SURFACE_SPOT_RANGE), metavar=("LO", "HI"),
                        help="PDE grid bounds as fractions of the initial spot")
    parser.add_argument("--oracle-paths", type=int, default=C.ORACLE_PATHS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-paths", type=int, default=None)
    parser.add_argument("--quad-grid", type=int, default=C.Q.DEFAULT_QUAD_GRID)
    parser.add_argument("--cost-bp", type=float, default=C.COST_BP)
    parser.add_argument("--maturity-months", type=int, default=C.Q.MATURITY_MONTHS)
    parser.add_argument("--lockout-months", type=int, default=C.Q.LOCKOUT_MONTHS)
    parser.add_argument("--disk-cache", action="store_true", help="share states through <out>/cache")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true", help="two cells, at most 8 check paths, 1 oracle path")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.cells = args.cells or [f"{C.MODELS[0]}:front", "term_flat_q:front"]
        args.check_paths, args.exact_paths = min(args.check_paths, 8), min(args.exact_paths, 8)
        args.oracle_paths = min(args.oracle_paths, 1)
    cells = parse_cells(args.cells)
    bootstrap, stress, paths_manifest = S01.load_paths(args.out_dir)
    terms = study_terms(bootstrap.dates, maturity_months=args.maturity_months, lockout_months=args.lockout_months)
    coupon = fair_coupon(bootstrap, terms, model=C.Q.REFERENCE_MODEL, quad_grid=args.quad_grid)
    product = C.Q.build_product(terms, float(bootstrap.spot[0, 0]), coupon.coupon)
    C.write_json(args.out_dir / "coupon.json", {"coupon": coupon.summary(), "terms": terms.summary(),
                                                "reference_model": C.Q.REFERENCE_MODEL})
    print(f"fair coupon {coupon.coupon:.4%} under {C.Q.REFERENCE_MODEL} (|PV| {abs(coupon.pv):,.0f}); "
          f"{len(cells)} cells x {bootstrap.n_paths} paths, provider {args.provider}, workers {args.workers}", flush=True)
    disk_dir = str(args.out_dir / "cache") if args.disk_cache else None
    runs: Dict[str, Any] = {}
    oracle = list(range(args.oracle_paths))
    common = dict(cost_bp=args.cost_bp, workers=args.workers, quad_grid=args.quad_grid, disk_dir=disk_dir,
                  spot_range=tuple(args.spot_range))
    for model, hedge in cells:
        cell = C.cell_name(model, hedge)
        plan = [(cell, bootstrap, args.provider, oracle),
                (f"{cell}__stress", stress, args.provider, [])]
        if args.check_paths:
            plan.append((f"{cell}__ladder_quad", bootstrap.take(range(min(args.check_paths, bootstrap.n_paths))),
                         "ladder", oracle[:1]))
        if args.exact_paths:
            plan.append((f"{cell}__exact_quad", bootstrap.take(range(min(args.exact_paths, bootstrap.n_paths))),
                         "exact", oracle[:1]))
        for name, batch, provider, checks in plan:
            config = cell_config(product, model, hedge, provider=provider,
                                 batch_paths=batch_for(batch.n_paths, args.workers, args.batch_paths), **common)
            run = run_cell(batch, config, args.out_dir / "cells" / name, resume=args.resume, oracle_paths=checks)
            runs[name] = run
            state = "skipped" if run["skipped"] else f"{run['seconds']:.0f}s"
            gate = run["gate"]
            oracle_state = ("–" if not run["oracle"] else "ok" if all(r["passed"] for r in run["oracle"]) else "FAIL")
            day0 = run.get("day0_book_mark_bp")
            print(f"  {name:32s} {state:>8s}  gate {gate['max_pv_gap_bp']:.2f} bp / {gate['max_delta_gap_hands']:.2f} hands "
                  f"({'ok' if gate['passed'] else 'FAIL, no results'})  oracle {oracle_state}  "
                  f"day0 {'–' if day0 is None else f'{day0:.2f} bp'}", flush=True)
    C.write_json(args.out_dir / "fleet_manifest.json", {
        "coupon": coupon.summary(), "terms": terms.summary(), "paths": paths_manifest, "runs": runs,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    })
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
