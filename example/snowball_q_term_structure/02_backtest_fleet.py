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
from quantark.asset.equity.param.engine_params import QUAD_READOUT_MODES  # noqa: E402
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
    engine = SnowballQuadEngine(params=QuadParams(
        grid_points=int(task["quad_grid"]),
        readout=str(task.get("quad_readout", C.DEFAULT_QUAD_READOUT)),
    ))
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


#: Risk-profile keys that change what the run PRICES or MEASURES, and so
#: belong in the fingerprint.  A display-only option must not appear here.
RISK_FINGERPRINT_KEYS = (
    "record_carry_exposure",
    "carry_audit_mode",
    "carry_audit_dates",
    "hedge_ratio",
    "futures_bump_points",
    "audit_spot_bump_rel",
    "identity_spot_bumps_rel",
    "audit_yield_bump",
    "delta_tolerance_hands",
    "rhoq_tolerance_bp",
    "stress_dates",
    "reference_notional",
    "exploratory_audit_override",
)


def fingerprint(task: Dict[str, Any]) -> str:
    keys = ("model", "hedge", "quad_grid", "cost_bp", "rate", "vol_tenor", "coupon",
            "delta_threshold", "round_contracts", "inception", "trade_end")
    payload = {k: task[k] for k in keys}
    payload["initial_price_mode"] = "traded_zero"
    # The readout changes prices, so a legacy cell must not resume into a run
    # that fixed it. Absent or default it contributes nothing, which keeps
    # every fingerprint banked before the mode existed valid.
    readout = task.get("quad_readout", C.DEFAULT_QUAD_READOUT)
    if readout != C.DEFAULT_QUAD_READOUT:
        payload["quad_readout"] = readout
    risk = task.get("risk") or {}
    # A legacy run carries no risk block at all, so its fingerprint is
    # unchanged and old resumes keep working.
    if risk:
        payload["risk"] = {k: risk.get(k) for k in RISK_FINGERPRINT_KEYS}
        payload["objective"] = C.BUCKET_OBJECTIVES.get(task["hedge"])
        payload["correction_pair"] = risk.get("correction_pair")
        # Absent or None it contributes nothing, which keeps every cell
        # banked before the hedge gap existed resumable. It only enters the
        # fingerprint when it is actually declared, because only then does
        # it add price calls and a measured column.
        if risk.get("hedge_resolution_rel") is not None:
            payload["hedge_resolution_rel"] = risk["hedge_resolution_rel"]
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _carry_settings(risk: Dict[str, Any]):
    """The task's resolved CarryRiskSettings, or None when nothing is on."""
    if not risk.get("record_carry_exposure") and risk.get(
        "carry_audit_mode", "none"
    ) == "none":
        return None
    from quantark.backtest.futures_risk import CarryRiskSettings

    return CarryRiskSettings(
        reference_notional=float(risk.get("reference_notional", C.NOTIONAL)),
        futures_bump_points=float(risk.get("futures_bump_points", 1.0)),
        audit_spot_bump_rel=risk.get("audit_spot_bump_rel"),
        identity_spot_bumps_rel=tuple(risk.get(
            "identity_spot_bumps_rel", CarryRiskSettings().identity_spot_bumps_rel
        )),
        hedge_resolution_rel=risk.get("hedge_resolution_rel"),
        audit_yield_bump=float(risk.get("audit_yield_bump", 1e-4)),
        delta_tolerance_hands=float(risk.get("delta_tolerance_hands", 0.01)),
        rhoq_tolerance_bp=float(risk.get("rhoq_tolerance_bp", 0.01)),
        stress_dates=tuple(
            datetime.fromisoformat(d) for d in risk.get("stress_dates", ())
        ),
    )


def run_cell(task: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _context(task["history_dir"], task["rate"], task["vol_tenor"])
    risk: Dict[str, Any] = dict(task.get("risk") or {})
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
        engine_config=C.engine_config_for(
            model,
            quad_grid_points=int(task["quad_grid"]),
            quad_readout=str(task.get("quad_readout", C.DEFAULT_QUAD_READOUT)),
        ),
        strategy=C.hedge_strategy_for(
            task["hedge"],
            delta_threshold=float(task["delta_threshold"]),
            round_contracts=bool(task["round_contracts"]),
            hedge_ratio=float(risk.get("hedge_ratio", 1.0)),
        ),
        roll_policy=C.hedge_roll_policy_for(task["hedge"]),
        dividend_roll_policy=(
            None
            if C.uses_buckets(task["hedge"])
            else C.dividend_roll_policy_for(C.Q_MODELS[task["model"]])
        ),
        record_carry_exposure=bool(risk.get("record_carry_exposure", False)),
        carry_audit_mode=str(risk.get("carry_audit_mode", "none")),
        carry_audit_dates=tuple(
            datetime.fromisoformat(d) for d in risk.get("carry_audit_dates", ())
        ),
        carry_risk_settings=_carry_settings(risk),
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
    engine = AutocallableBacktestEngine(config)
    results = engine.run()
    elapsed = time.perf_counter() - started
    # Measured wall clock and price counts by stage, so a fleet runtime
    # estimate comes from a measurement rather than an arithmetic guess.
    carry_cost = engine._inner.carry_cost()
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
        "quad_readout": task.get("quad_readout", C.DEFAULT_QUAD_READOUT),
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
        "carry_cost": carry_cost,
    }
    summary.update({k: v for k, v in results.get_summary().items() if k not in summary})
    run_dir = C.run_dir_for(Path(task["out_dir"]), task["inception_tag"], task["model"], task["hedge"])
    run_format = C.RUN_FORMAT_LEGACY
    run_config = None
    audits = None
    if risk:
        run_format = C.RUN_FORMAT_CARRY
        audits = C.audit_coverage(
            C.result_frame(results, "hedge_legs"),
            C.result_frame(results, "hedge_attribution"),
        )
        run_config = {
            "format_version": run_format,
            "objective": C.BUCKET_OBJECTIVES.get(task["hedge"]),
            "hedge_policy": task["hedge"],
            "roll_selector": C.HEDGE_ROLL_SELECTOR[task["hedge"]],
            "q_model": task["model"],
            "dividend_source": C.Q_MODELS[task["model"]].dividend_source,
            "extrapolation": C.Q_MODELS[task["model"]].extrapolation,
            "min_tenor_days": C.Q_MODELS[task["model"]].min_tenor_days,
            "quad_grid": task["quad_grid"],
            "quad_readout": task.get("quad_readout", C.DEFAULT_QUAD_READOUT),
            "delta_threshold": task["delta_threshold"],
            "round_contracts": task["round_contracts"],
            "cost_bp": task["cost_bp"],
            "rate": task["rate"],
            "risk": dict(risk),
            "fingerprint": summary["fingerprint"],
            "source_digest": source_digest(),
        }
        summary["audit_coverage"] = audits["audit_coverage"]
        summary["all_measured_audits_passed"] = audits["all_measured_passed"]
        summary["exploratory_audit_override"] = bool(
            risk.get("exploratory_audit_override")
        )
    C.write_run(
        run_dir,
        results,
        summary,
        run_format=run_format,
        run_config=run_config,
        audit_summary=audits,
    )
    return summary


def error_category(exc: Exception) -> str:
    """Coarse classification, so a data gap is never retried like a bug.

    A missing price, an infeasible hedge and a numeric audit failure are
    distinct outcomes with distinct responses; collapsing them would let a
    failed objective be retried identically until it looked like flakiness.
    """
    text = str(exc)
    if "infeasible" in text or "two distinct futures tenors" in text:
        return "infeasible_hedge"
    if (
        "mark" in text
        or "no contract with at least" in text
        or "found no contract" in text
    ):
        return "missing_price"
    if "audit" in text or "tolerance" in text:
        return "numeric_audit"
    if isinstance(exc, C.StudyDataError):
        return "study_data"
    return "other"


def _run_cell_guarded(task: Dict[str, Any]) -> Dict[str, Any]:
    try:
        summary = run_cell(task)
        return {"status": "ok", "task": task, "summary": summary}
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest, fail-closed per cell
        failure = {
            "status": "failed",
            "task": task,
            "date": datetime.now().isoformat(timespec="seconds"),
            "objective": C.BUCKET_OBJECTIVES.get(task["hedge"]),
            "input_fingerprint": fingerprint(task),
            "error_category": error_category(exc),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        run_dir = C.run_dir_for(
            Path(task["out_dir"]), task["inception_tag"], task["model"], task["hedge"]
        )
        # A failure NEVER replaces an earlier completed run: it is written
        # beside it, and the completed summary keeps its own file.
        run_dir.mkdir(parents=True, exist_ok=True)
        C.write_json(run_dir / "failure.json", failure)
        return failure


# ---------------------------------------------------------------------------
# Fleet orchestration
# ---------------------------------------------------------------------------


def parse_cells(values: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for v in values:
        if ":" not in v:
            raise SystemExit(f"--cells entries look like model:hedge, got {v!r}")
        model, hedge = v.split(":", 1)
        if model not in C.Q_MODELS or hedge not in C.HEDGE_ROLL_SELECTOR:
            raise SystemExit(f"unknown cell {v!r}")
        out.append((model, hedge))
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history-dir", type=Path, default=C.DEFAULT_HISTORY_DIR)
    parser.add_argument("--out-dir", type=Path, default=C.DEFAULT_OUT_DIR)
    parser.add_argument("--cells", nargs="+", default=None, help="model:hedge, overrides --study-grid")
    parser.add_argument(
        "--study-grid",
        choices=("legacy", "buckets", "all"),
        default="legacy",
        help=(
            "legacy = the original cells (default, so old commands are "
            "unchanged); buckets = the revised 14-cell primary grid; "
            "all = both"
        ),
    )
    parser.add_argument("--quick", action="store_true", help=f"{QUICK_INCEPTIONS} inceptions (legacy grid: {len(QUICK_CELLS)} cells)")
    parser.add_argument("--max-inceptions", type=int, default=None)
    parser.add_argument("--first-month", default=f"{C.FIRST_INCEPTION_MONTH[0]}-{C.FIRST_INCEPTION_MONTH[1]:02d}")
    parser.add_argument("--include-censored", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quad-grid", type=int, default=C.DEFAULT_QUAD_GRID)
    parser.add_argument(
        "--quad-readout",
        choices=QUAD_READOUT_MODES,
        default=C.DEFAULT_QUAD_READOUT,
        help=(
            "how the engine recovers the price from its nodal surface; "
            "'transition' removes the delta staircase but moves prices "
            "(docs/bucket-futures-hedge/quad-readout/)"
        ),
    )
    parser.add_argument("--cost-bp", type=float, default=DEFAULT_COST_BP)
    parser.add_argument("--delta-threshold", type=float, default=0.0, help="contracts band before rebalancing")
    parser.add_argument("--no-round-contracts", action="store_true")
    parser.add_argument("--rate", type=float, default=C.FLAT_RATE)
    parser.add_argument("--vol-tenor", type=float, default=C.ATM_VOL_TENOR_YEARS)
    risk = parser.add_argument_group("carry risk recording")
    risk.add_argument("--record-carry-exposure", action="store_true")
    risk.add_argument(
        "--carry-audit-mode", choices=("none", "sampled", "daily"), default=None,
        help="default: daily under --study-grid buckets, none otherwise",
    )
    risk.add_argument("--carry-audit-dates", nargs="+", default=())
    risk.add_argument("--hedge-ratio", type=float, default=1.0)
    risk.add_argument("--futures-bump-points", type=float, default=1.0)
    risk.add_argument("--audit-spot-bump-rel", type=float, default=None)
    from quantark.backtest.futures_risk import CarryRiskSettings
    risk.add_argument(
        "--identity-spot-bumps-rel", type=float, nargs="+",
        default=CarryRiskSettings().identity_spot_bumps_rel,
        help="descending matched spot-bump ladder for the chain identity (at least 3)",
    )
    risk.add_argument(
        "--hedge-resolution-rel", type=float, default=None,
        help="desk rebalance resolution, e.g. 0.0025; opt-in, adds 2 price "
             "calls a day and reports pricing_delta_hedge_gap_hands",
    )
    risk.add_argument("--audit-yield-bump", type=float, default=1e-4)
    risk.add_argument("--delta-tolerance-hands", type=float, default=0.01)
    risk.add_argument("--rhoq-tolerance-bp", type=float, default=0.01)
    risk.add_argument("--stress-dates", nargs="+", default=())
    return parser.parse_args(argv)


def resolve_cells(args) -> List[Tuple[str, str]]:
    """Explicit --cells wins; otherwise the named grid."""
    if args.cells:
        return parse_cells(args.cells)
    if args.study_grid == "buckets":
        return list(C.PRIMARY_BUCKET_CELLS)
    legacy = list(QUICK_CELLS if args.quick else DEFAULT_CELLS)
    if args.study_grid == "all":
        seen = list(legacy)
        for cell in C.PRIMARY_BUCKET_CELLS:
            if cell not in seen:
                seen.append(cell)
        return seen
    return legacy


def resolve_risk_profile(args) -> Dict[str, Any]:
    """The run's resolved recording settings, and whether they are a certificate.

    The buckets profile records exposure and audits daily unless the caller
    asks otherwise.  Such an override is fingerprinted and marked, because a
    run that skipped audits cannot later be read as a full daily-audit
    certificate.
    """
    buckets = args.study_grid == "buckets" or any(
        C.uses_buckets(hedge) for _, hedge in resolve_cells(args)
    )
    requested = args.carry_audit_mode
    mode = requested if requested is not None else ("daily" if buckets else "none")
    exploratory = bool(buckets and requested is not None and requested != "daily")
    return {
        "record_carry_exposure": bool(args.record_carry_exposure or buckets),
        "carry_audit_mode": mode,
        "carry_audit_dates": [str(d) for d in args.carry_audit_dates],
        "hedge_ratio": float(args.hedge_ratio),
        "futures_bump_points": float(args.futures_bump_points),
        "audit_spot_bump_rel": args.audit_spot_bump_rel,
        "identity_spot_bumps_rel": list(args.identity_spot_bumps_rel),
        "hedge_resolution_rel": args.hedge_resolution_rel,
        "audit_yield_bump": float(args.audit_yield_bump),
        "delta_tolerance_hands": float(args.delta_tolerance_hands),
        "rhoq_tolerance_bp": float(args.rhoq_tolerance_bp),
        "stress_dates": [str(d) for d in args.stress_dates],
        "reference_notional": C.NOTIONAL,
        "exploratory_audit_override": exploratory,
    }


def resumable(existing: Dict[str, Any], task: Dict[str, Any], run_dir: Path) -> bool:
    """Whether a stored run can stand in for this task.

    It needs the same fingerprint, a completed status, the right format and
    every file that format requires.  A legacy summary can never satisfy a
    task that asked for audits: it has neither the frames nor the coverage.
    """
    if existing.get("fingerprint") != fingerprint(task):
        return False
    if existing.get("status") == "failed":
        return False
    wanted = C.RUN_FORMAT_CARRY if task.get("risk") else C.RUN_FORMAT_LEGACY
    if existing.get("run_format", C.RUN_FORMAT_LEGACY) != wanted:
        return False
    for name in C.required_frames(wanted):
        if not (run_dir / f"{name}.csv").exists():
            return False
    if wanted == C.RUN_FORMAT_CARRY:
        for name in ("run_config.json", "audit_summary.json"):
            if not (run_dir / name).exists():
                return False
        requested = (task.get("risk") or {}).get("carry_audit_mode", "none")
        if requested != "none" and existing.get("audit_coverage") != "measured":
            return False
    return True


def source_digest() -> str:
    """Content digest of the source that decides pricing, curve and sizing.

    A revision identifier alone is insufficient: the working tree routinely
    carries uncommitted changes, and two runs of "the same revision" can
    price differently.
    """
    import hashlib as _hashlib
    from pathlib import Path as _Path

    import quantark.backtest.futures_risk as _risk
    import quantark.backtest.replay.carry_context as _ctx
    import quantark.backtest.replay.carry_recorder as _rec
    import quantark.backtest.replay.carry_risk as _cr
    import quantark.backtest.replay.carry_stress as _cs
    import quantark.backtest.replay.engine as _engine
    import quantark.backtest.strategy.futures_bucket_strategy as _bucket

    roots = [_Path(C.__file__), _Path(__file__)]
    roots += [
        _Path(module.__file__)
        for module in (_risk, _ctx, _cr, _cs, _rec, _engine, _bucket)
    ]
    digest = _hashlib.sha256()
    for path in sorted(roots):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cells = resolve_cells(args)
    risk_profile = resolve_risk_profile(args)
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
    coupon_key = (
        f"grid={args.quad_grid}|rate={args.rate}|vol_tenor={args.vol_tenor}"
        f"|readout={args.quad_readout}"
    )
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
                "quad_readout": args.quad_readout,
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
                "quad_grid": args.quad_grid, "quad_readout": args.quad_readout,
                "cost_bp": args.cost_bp, "rate": args.rate,
                "vol_tenor": args.vol_tenor, "delta_threshold": args.delta_threshold,
                "round_contracts": not args.no_round_contracts,
                "history_dir": str(args.history_dir), "out_dir": str(out_dir),
            }
            if risk_profile["record_carry_exposure"] or risk_profile[
                "carry_audit_mode"
            ] != "none":
                task["risk"] = dict(risk_profile)
            run_dir = C.run_dir_for(out_dir, s.tag, model, hedge)
            existing = load_json(run_dir / "run_summary.json") if args.resume else None
            if existing and resumable(existing, task, run_dir):
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
                "quad_readout": args.quad_readout,
                "cost_bp": args.cost_bp, "rate": args.rate, "vol_tenor": args.vol_tenor,
                "delta_threshold": args.delta_threshold, "round_contracts": not args.no_round_contracts,
                "study_grid": args.study_grid, "risk": resolve_risk_profile(args),
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
