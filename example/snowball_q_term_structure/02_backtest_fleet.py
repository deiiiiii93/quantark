"""Stage 02 - Snowball hedging backtest fleet: q models x hedge contracts.

For every monthly inception the SAME 1Y snowball (fair coupon solved once
under the reference model, ``term_flat_q``) is sold and delta-hedged with IM
futures through ``quantark.backtest.replay``, once per cell:

    cell = (q model, hedge contract policy)

    q model             what the pricer receives as dividend/carry each day
    -----------------   -------------------------------------------------------
    flat_from_hedge     the engine default: flat yield implied from the
                        contract the hedge currently holds (simple
                        compounding, floored at zero)
    flat_from_far       the same flat channel, always inverted from the
                        LONGEST listed contract whatever the hedge holds
    term_flat_q         the whole IM chain as q(T), endpoint zero yield held
    term_flat_fwd       the whole IM chain as q(T), forward carry held
    surface_fwd         MO option-implied forwards (cross-market control)
    term_opt_tail       the chain in log-forward space, option-forward tail

    hedge policy        which contract carries the delta hedge
    -----------------   -------------------------------------------------------
    front               front month, rolled 5 days before expiry
    far                 longest listed contract, rolled 5 days before expiry

The two fields are independent for the four term models: they read the whole
chain, so ``__front`` and ``__far`` differ only in the hedge leg.  They are
NOT independent for ``flat_from_hedge``: the engine inverts whichever
contract the roll policy holds (``engine.py`` passes one ``selected`` row to
both the hedge trade and ``build_env``), so ``flat_from_hedge__far`` prices
off the longest listed contract and ``flat_from_hedge__front`` off the front
month.  The name says so; the paired difference of that model therefore mixes
a carry change with a hedge change, and only the term pairs isolate the hedge.

Every cell of one inception shares the contract, the spot path, the vol
channel (ATM 1Y IV off the admitted MO surface), the rate and the cost
model, so paired differences are attributable to the q model / hedge
contract alone.  Each run replays from inception to knock-out, maturity
settlement, or the data end (``censored``).

Accounting: the contract is traded at zero PV (fair coupon), so every cell
starts from ``initial_product_price = 0`` rather than from its own model's
inception price.  A model that misprices the contract shows that as a
day-one mark in its P&L path; at settlement the payoff is what it is, so the
paired terminal P&L difference is exactly the hedge P&L difference.

Outputs under ``--out-dir`` (locally excluded from git):

    runs/<inception>/<model>__<hedge>/{states,greeks,trades,rebalances,actions}.csv
                                     /run_summary.json
    inceptions.json      solved terms + fair coupon per inception
    fleet_manifest.json  configuration, cells, per-run status and timing

Each cell writes its summary as it finishes; ``--resume`` keeps completed
cells whose fingerprint (grid, cost, rate, coupon) matches.

Run:
    .venv/bin/python example/snowball_q_term_structure/02_backtest_fleet.py --quick
    nohup caffeinate -i -m -s .venv/bin/python \\
        example/snowball_q_term_structure/02_backtest_fleet.py --workers 4 > fleet.log 2>&1 &
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine  # noqa: E402
from quantark.asset.equity.param import QuadParams  # noqa: E402
from quantark.backtest.replay import (  # noqa: E402
    AutocallableBacktestConfig,
    AutocallableBacktestEngine,
    AutocallableDeltaHedgeStrategy,
)
from quantark.backtest.transaction_costs import ProportionalCostModel, ZeroCostModel  # noqa: E402
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402

DEFAULT_CELLS: Tuple[Tuple[str, str], ...] = (
    ("flat_from_hedge", "front"),
    ("term_flat_q", "front"),
    ("term_flat_fwd", "front"),
    ("surface_fwd", "front"),
    ("term_opt_tail", "front"),
    ("flat_from_hedge", "far"),
    ("term_flat_q", "far"),
    # Carry off the longest listed contract, delta in the front month: the
    # baseline's hedge leg, so their paired difference is the carry contract.
    ("flat_from_far", "front"),
)
QUICK_CELLS: Tuple[Tuple[str, str], ...] = (
    ("flat_from_hedge", "front"),
    ("term_flat_q", "front"),
    ("flat_from_hedge", "far"),
)
QUICK_INCEPTIONS = 4
DEFAULT_COST_BP = 1.0  # all-in proportional cost per side, bp of traded notional


# ---------------------------------------------------------------------------
# Per-process context (loaded once per worker)
# ---------------------------------------------------------------------------

_CTX: Dict[str, Any] = {}


def _context(history_dir: str, rate: float, vol_tenor: float) -> Dict[str, Any]:
    key = f"{history_dir}|{rate}|{vol_tenor}"
    ctx = _CTX.get(key)
    if ctx is None:
        frames = C.load_history(history_dir)
        history = C.surface_history(history_dir)
        dataset = C.build_market_dataset(frames, history=history, rate=rate, vol_tenor_years=vol_tenor)
        calendar = C.TradingCalendar.from_frames(frames)
        ctx = {"frames": frames, "history": history, "dataset": dataset, "calendar": calendar}
        _CTX[key] = ctx
    return ctx


# ---------------------------------------------------------------------------
# Inceptions and fair coupons
# ---------------------------------------------------------------------------


def inception_env(ctx: Dict[str, Any], inception: date, rate: float) -> PricingEnvironment:
    """Day-one env of the reference model, exactly as its replay builds it."""
    ts = pd.Timestamp(inception)
    market = ctx["dataset"].get_market_row(ts)
    chain = ctx["dataset"].get_futures_slice(ts)
    div = C.dividend_for(
        C.Q_MODELS[C.REFERENCE_MODEL], valuation=ts, spot=float(market["spot"]),
        rate=rate, chain_slice=chain,
    )
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=float(market["spot"]), asset_name=C.UNDERLYING_NAME),
        vol_surface=FlatVolSurface(volatility=float(market["volatility"])),
        rate_curve=FlatRateCurve(rate=float(rate)),
        div_yield=div,
        valuation_date=ts.to_pydatetime(),
    )


def solve_inception(task: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _context(task["history_dir"], task["rate"], task["vol_tenor"])
    inception = date.fromisoformat(task["inception"])
    terms = C.build_terms(inception, ctx["calendar"])
    env = inception_env(ctx, inception, task["rate"])
    s0 = float(env.spot_quote.spot)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=int(task["quad_grid"])))
    started = time.perf_counter()
    solution = C.solve_fair_coupon(lambda c: engine.price(C.build_product(terms, s0, c), env))
    return {
        "inception": inception.isoformat(),
        "s0": s0,
        "atm_vol": float(env.vol_surface.get_vol(0.0, 1.0, 0.0)),
        "q_at_maturity_reference": float(env.div_yield.get_yield(terms.maturity_years)),
        "coupon": solution.coupon,
        "coupon_solution": solution.summary(),
        "terms": terms.summary(),
        "seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


def fingerprint(task: Dict[str, Any]) -> str:
    keys = ("model", "hedge", "quad_grid", "cost_bp", "rate", "vol_tenor", "coupon",
            "delta_threshold", "round_contracts", "inception", "trade_end")
    payload = {k: task[k] for k in keys}
    payload["initial_price_mode"] = "traded_zero"
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def run_cell(task: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _context(task["history_dir"], task["rate"], task["vol_tenor"])
    inception = date.fromisoformat(task["inception"])
    terms = C.build_terms(inception, ctx["calendar"])
    model = C.Q_MODELS[task["model"]]
    product = C.build_product(terms, task["s0"], task["coupon"])
    cost_model = (
        ProportionalCostModel(commission_rate=float(task["cost_bp"]) * 1e-4)
        if task["cost_bp"] > 0
        else ZeroCostModel()
    )
    config = AutocallableBacktestConfig(
        product=product,
        market_data=ctx["dataset"],
        engine_config=C.engine_config_for(model, quad_grid_points=int(task["quad_grid"])),
        strategy=AutocallableDeltaHedgeStrategy(
            delta_threshold=float(task["delta_threshold"]),
            round_contracts=bool(task["round_contracts"]),
        ),
        roll_policy=C.HEDGE_POLICIES[task["hedge"]](),
        dividend_roll_policy=C.dividend_roll_policy_for(C.Q_MODELS[task["model"]]),
        transaction_cost_model=cost_model,
        product_quantity=C.PRODUCT_QUANTITY,
        underlying=C.UNDERLYING_NAME,
        initial_product_price=0.0,  # traded at zero PV: every cell starts from the same book value
        start_date=datetime.combine(inception, datetime.min.time()),
        end_date=datetime.combine(date.fromisoformat(task["trade_end"]), datetime.min.time()),
        calculate_surfaces=False,
        calculate_event_probabilities=False,
        terminate_on_lifecycle_end=True,
        metadata={"study": "snowball_q_term_structure", "model": task["model"], "hedge": task["hedge"]},
    )
    started = time.perf_counter()
    results = AutocallableBacktestEngine(config).run()
    elapsed = time.perf_counter() - started
    states = results.states_df
    actions = results.actions_df
    summary: Dict[str, Any] = {
        "schema_version": C.SCHEMA_VERSION,
        "fingerprint": fingerprint(task),
        "inception": task["inception"],
        "model": task["model"],
        "hedge": task["hedge"],
        "coupon": task["coupon"],
        "s0": task["s0"],
        "notional": C.NOTIONAL,
        "quad_grid": task["quad_grid"],
        "cost_bp": task["cost_bp"],
        "rate": task["rate"],
        "censored_schedule": task["censored"],
        "elapsed_seconds": elapsed,
        "final_total_pnl": float(states["total_pnl"].iloc[-1]),
        "final_product_pnl": float(states["product_pnl"].iloc[-1]),
        "final_hedge_pnl": float(states["hedge_pnl"].iloc[-1]),
        "transaction_costs": float(states["transaction_costs"].iloc[-1]),
        "knocked_in": bool(states["knocked_in"].iloc[-1]),
        "knocked_out": bool(states["knocked_out"].iloc[-1]),
        "matured": bool(states["matured"].iloc[-1]),
        "n_actions": int(len(actions)),
    }
    summary.update({k: v for k, v in results.get_summary().items() if k not in summary})
    run_dir = C.run_dir_for(Path(task["out_dir"]), task["inception_tag"], task["model"], task["hedge"])
    C.write_run(run_dir, results, summary)
    return summary


def _run_cell_guarded(task: Dict[str, Any]) -> Dict[str, Any]:
    try:
        summary = run_cell(task)
        return {"status": "ok", "task": task, "summary": summary}
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest, fail-closed per cell
        return {
            "status": "failed",
            "task": task,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Fleet orchestration
# ---------------------------------------------------------------------------


def parse_cells(values: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for v in values:
        if ":" not in v:
            raise SystemExit(f"--cells entries look like model:hedge, got {v!r}")
        model, hedge = v.split(":", 1)
        if model not in C.Q_MODELS or hedge not in C.HEDGE_POLICIES:
            raise SystemExit(f"unknown cell {v!r}")
        out.append((model, hedge))
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history-dir", type=Path, default=C.DEFAULT_HISTORY_DIR)
    parser.add_argument("--out-dir", type=Path, default=C.DEFAULT_OUT_DIR)
    parser.add_argument("--cells", nargs="+", default=None, help="model:hedge, default = the study grid")
    parser.add_argument("--quick", action="store_true", help=f"{QUICK_INCEPTIONS} inceptions x {len(QUICK_CELLS)} cells")
    parser.add_argument("--max-inceptions", type=int, default=None)
    parser.add_argument("--first-month", default=f"{C.FIRST_INCEPTION_MONTH[0]}-{C.FIRST_INCEPTION_MONTH[1]:02d}")
    parser.add_argument("--include-censored", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quad-grid", type=int, default=C.DEFAULT_QUAD_GRID)
    parser.add_argument("--cost-bp", type=float, default=DEFAULT_COST_BP)
    parser.add_argument("--delta-threshold", type=float, default=0.0, help="contracts band before rebalancing")
    parser.add_argument("--no-round-contracts", action="store_true")
    parser.add_argument("--rate", type=float, default=C.FLAT_RATE)
    parser.add_argument("--vol-tenor", type=float, default=C.ATM_VOL_TENOR_YEARS)
    return parser.parse_args(argv)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cells = parse_cells(args.cells) if args.cells else list(QUICK_CELLS if args.quick else DEFAULT_CELLS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y, m = (int(x) for x in args.first_month.split("-"))

    ctx = _context(str(args.history_dir), args.rate, args.vol_tenor)
    calendar = ctx["calendar"]
    schedules = C.enumerate_inceptions(calendar, first_month=(y, m))
    if not args.include_censored:
        schedules = [s for s in schedules if not s.censored]
    limit = QUICK_INCEPTIONS if args.quick and args.max_inceptions is None else args.max_inceptions
    if limit:
        schedules = schedules[:limit]
    if not schedules:
        raise SystemExit("no inceptions to run")
    print(
        f"fleet: {len(schedules)} inceptions ({schedules[0].inception} .. {schedules[-1].inception}) "
        f"x {len(cells)} cells, quad_grid={args.quad_grid}, cost={args.cost_bp} bp, workers={args.workers}",
        flush=True,
    )

    # --- fair coupons (cached; the contract is shared by every cell) ------
    inceptions_path = out_dir / "inceptions.json"
    cached = load_json(inceptions_path) if args.resume else None
    coupon_key = f"grid={args.quad_grid}|rate={args.rate}|vol_tenor={args.vol_tenor}"
    inception_records: Dict[str, Dict[str, Any]] = {}
    if cached and cached.get("coupon_key") == coupon_key:
        inception_records = dict(cached.get("inceptions", {}))
    todo = [s for s in schedules if s.inception.isoformat() not in inception_records]
    if todo:
        started = time.perf_counter()
        tasks = [
            {
                "inception": s.inception.isoformat(), "history_dir": str(args.history_dir),
                "rate": args.rate, "vol_tenor": args.vol_tenor, "quad_grid": args.quad_grid,
            }
            for s in todo
        ]
        for rec in _map(solve_inception, tasks, args.workers):
            inception_records[rec["inception"]] = rec
            print(
                f"  coupon {rec['inception']}: {rec['coupon']:.4%} (s0={rec['s0']:,.1f}, "
                f"vol={rec['atm_vol']:.3f}, q_ref(T)={rec['q_at_maturity_reference']:+.2%}, "
                f"{rec['coupon_solution']['evaluations']} prices)",
                flush=True,
            )
        C.write_json(inceptions_path, {"coupon_key": coupon_key, "inceptions": inception_records})
        print(f"coupons solved in {time.perf_counter() - started:.0f}s", flush=True)

    # --- cells --------------------------------------------------------------
    tasks: List[Dict[str, Any]] = []
    manifest_runs: List[Dict[str, Any]] = []
    for s in schedules:
        rec = inception_records[s.inception.isoformat()]
        for model, hedge in cells:
            task = {
                "inception": s.inception.isoformat(), "inception_tag": s.tag,
                "trade_end": s.trade_end.isoformat(), "censored": s.censored,
                "model": model, "hedge": hedge,
                "coupon": rec["coupon"], "s0": rec["s0"],
                "quad_grid": args.quad_grid, "cost_bp": args.cost_bp, "rate": args.rate,
                "vol_tenor": args.vol_tenor, "delta_threshold": args.delta_threshold,
                "round_contracts": not args.no_round_contracts,
                "history_dir": str(args.history_dir), "out_dir": str(out_dir),
            }
            run_dir = C.run_dir_for(out_dir, s.tag, model, hedge)
            existing = load_json(run_dir / "run_summary.json") if args.resume else None
            if existing and existing.get("fingerprint") == fingerprint(task):
                manifest_runs.append({"status": "resumed", "task": task, "summary": existing})
                continue
            tasks.append(task)
    print(f"{len(tasks)} cells to run, {len(manifest_runs)} resumed", flush=True)

    started = time.perf_counter()
    done = 0
    for outcome in _map(_run_cell_guarded, tasks, args.workers):
        done += 1
        manifest_runs.append(outcome)
        t = outcome["task"]
        if outcome["status"] == "ok":
            s = outcome["summary"]
            print(
                f"  [{done}/{len(tasks)}] {t['inception']} {C.cell_name(t['model'], t['hedge']):24s} "
                f"{s['termination_reason']:12s} {s['days_replayed']:4d}d  "
                f"pnl {s['final_total_pnl'] / C.NOTIONAL * 1e4:+8.1f} bp  "
                f"({s['elapsed_seconds']:.0f}s)",
                flush=True,
            )
        else:
            print(f"  [{done}/{len(tasks)}] {t['inception']} {C.cell_name(t['model'], t['hedge'])} FAILED: {outcome['error']}", flush=True)
        _write_manifest(out_dir, args, cells, schedules, manifest_runs, started)
    _write_manifest(out_dir, args, cells, schedules, manifest_runs, started)
    failed = [r for r in manifest_runs if r["status"] == "failed"]
    print(
        f"fleet done: {len(manifest_runs) - len(failed)} ok, {len(failed)} failed, "
        f"{time.perf_counter() - started:.0f}s",
        flush=True,
    )
    return 1 if failed else 0


def _write_manifest(out_dir, args, cells, schedules, runs, started) -> None:
    C.write_json(
        out_dir / "fleet_manifest.json",
        {
            "schema_version": C.SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "history_dir": str(args.history_dir), "quad_grid": args.quad_grid,
                "cost_bp": args.cost_bp, "rate": args.rate, "vol_tenor": args.vol_tenor,
                "delta_threshold": args.delta_threshold, "round_contracts": not args.no_round_contracts,
                "reference_model": C.REFERENCE_MODEL, "baseline_model": C.BASELINE_MODEL,
                "notional": C.NOTIONAL, "maturity_months": C.MATURITY_MONTHS,
                "lockout_months": C.LOCKOUT_MONTHS, "ko_pct": C.KO_PCT, "ki_pct": C.KI_PCT,
            },
            "cells": [C.cell_name(m, h) for m, h in cells],
            "models": {m: C.Q_MODELS[m].summary() for m, _ in cells},
            "inceptions": [s.summary() for s in schedules],
            "runs": [
                {k: v for k, v in r.items() if k != "traceback"} | {"traceback": r.get("traceback")}
                for r in runs
            ],
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def _map(fn, tasks, workers: int):
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield fn(task)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(fn, task) for task in tasks]
        for fut in as_completed(futures):
            yield fut.result()


if __name__ == "__main__":
    raise SystemExit(main())
