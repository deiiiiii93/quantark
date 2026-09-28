"""Re-run the carry identity audit over the fleet, with whole-cell alignment on.

The identity is holdings-free -- verified against the banked output, 45,348
paired rows across hedge policies, zero differing -- so the distinct work is
the (inception, model, date) states, not the 406 cells. 7,616 instead of
52,906, and every cell of a state would have reproduced the same number.

Uses the production audit primitives rather than re-deriving them:
``measure_product_carry_risk`` builds the buckets the identity must retain,
and ``sample_chain_identity`` walks the matched refinement ladder. The gate
is the one the recorder applies, ``|R| + E <= delta_tolerance_hands``, where
E is the change over the last rung of the ladder.

Usage:
    audit_rerun.py <out.csv> [--stretch 0.02 | --off] [--limit N] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date as date_cls
from pathlib import Path

REPO = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge")
STUDY = REPO / "example/snowball_q_term_structure"
RUNS = STUDY / "data/bucket_hedge_v2/gate_e/runs"
HIST = Path("/Users/fuxinyao/quant-ark/example/mo_volmodels/data/history")


def _bootstrap():
    """Resolve quantark to the WORKTREE in this process, parent or worker."""
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"{REPO}{os.pathsep}{existing}" if existing else str(REPO)
    )
    for key in [k for k in list(sys.modules)
                if k == "quantark" or k.startswith("quantark.")]:
        del sys.modules[key]
    for path in (str(STUDY), str(REPO)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


_bootstrap()

import pandas as pd  # noqa: E402
import quantark  # noqa: E402

assert str(REPO) in quantark.__file__, quantark.__file__

import _common as C  # noqa: E402
from quantark.asset.equity.engine.quad.snowball_quad_engine import (  # noqa: E402
    SnowballQuadEngine,
)
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker  # noqa: E402
from quantark.asset.equity.market import IndexFuturesQuote  # noqa: E402
from quantark.asset.equity.param import QuadParams  # noqa: E402
from quantark.backtest.futures_risk import CarryRiskSettings  # noqa: E402
from quantark.backtest.replay.carry_context import CarryCurveContext  # noqa: E402
from quantark.backtest.replay.carry_risk import (  # noqa: E402
    aggregate_book_risk,
    measure_product_carry_risk,
    sample_chain_identity,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402

_HISTORY = None


def history():
    global _HISTORY
    if _HISTORY is None:
        _HISTORY = C.load_history(HIST)
    return _HISTORY


def representative_cells():
    """One cell per (inception, model): the rest audit identically."""
    seen = {}
    for inception in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        for cell in sorted(p for p in inception.iterdir() if p.is_dir()):
            if not (cell / "hedge_attribution.csv").exists():
                continue
            model = cell.name.split("__", 1)[0]
            seen.setdefault((inception.name, model), cell)
    return seen


def settings_from(config) -> CarryRiskSettings:
    risk = config["risk"]
    settings = CarryRiskSettings(
        reference_notional=float(risk["reference_notional"]),
        delta_tolerance_hands=float(risk["delta_tolerance_hands"]),
        rhoq_tolerance_bp=float(risk["rhoq_tolerance_bp"]),
        futures_bump_points=float(risk["futures_bump_points"]),
        audit_yield_bump=float(risk["audit_yield_bump"]),
        identity_spot_bumps_rel=tuple(risk["identity_spot_bumps_rel"]),
        hedge_resolution_rel=risk.get("hedge_resolution_rel"),
    )
    # audit_spot_bump_rel is recorded null: the replay resolves it from the
    # run's pricing bump. It feeds the legacy 1% diagnostic only --
    # sample_chain_identity walks identity_spot_bumps_rel and never reads it
    # -- so any resolved value leaves R and E untouched. require_resolved
    # still demands one.
    return settings.resolved(
        reference_notional=float(risk["reference_notional"]),
        audit_spot_bump_rel=0.01,
    )


def audit_group(key, cell_path: str, stretch):
    """Every measured date of one (inception, model), on one worker."""
    inception, model = key
    cell = Path(cell_path)
    summary = json.loads((cell / "run_summary.json").read_text())
    config = json.loads((cell / "run_config.json").read_text())
    states = pd.read_csv(cell / "states.csv").set_index("date")
    legs = pd.read_csv(cell / "hedge_legs.csv")
    banked = pd.read_csv(cell / "hedge_attribution.csv").set_index("date")
    settings = settings_from(config)

    terms = C.build_terms(
        date_cls.fromisoformat(summary["inception"]),
        C.TradingCalendar.from_frames(history()),
    )
    original = C.build_product(terms, summary["s0"], summary["coupon"])
    node_legs = legs[legs.is_curve_node]

    quad = {"grid_points": config["quad_grid"],
            "readout": config.get("quad_readout", "legacy_linear")}
    if stretch is not None:
        quad["align_cell_stretch"] = float(stretch)

    rows = []
    dates = banked.index[banked["identity_residual_hands"].notna()]
    for day in dates:
        state = states.loc[day]
        quotes = tuple(
            IndexFuturesQuote(
                contract=r["contract"], maturity=float(r["tenor_years"]),
                price=float(r["price"]), multiplier=float(r["multiplier"]),
            )
            for r in node_legs[node_legs.date == day].to_dict("records")
        )
        if not quotes:
            continue
        context = CarryCurveContext(
            quotes, float(state.spot), FlatRateCurve(rate=float(state.rate)),
            config["extrapolation"], "CSI1000", pd.Timestamp(day),
        )

        def env(spot, dividend):
            return PricingEnvironment(
                spot_quote=SpotQuote(spot=spot),
                vol_surface=FlatVolSurface(volatility=float(state.volatility)),
                rate_curve=context.rate_curve, div_yield=dividend,
                valuation_date=pd.Timestamp(day).to_pydatetime(),
            )

        tracker = AutocallableLifecycleTracker(
            product=original, quantity=-1.0,
            start_date=pd.Timestamp(summary["inception"]),
        )
        tracker.lifecycle.knocked_in = bool(state.knocked_in)
        product = tracker.product_for_pricing(
            pd.Timestamp(day), env(context.spot, context.dividend())
        )
        engine = SnowballQuadEngine(QuadParams(**quad))

        def price_at(spot, dividend):
            return -engine.price(product, env(spot, dividend))

        try:
            product_risk = measure_product_carry_risk(
                price_at, context, delta_q=0.0,
                points=settings.futures_bump_points,
            )
            risk = aggregate_book_risk([(1.0, product_risk)])
            samples = sample_chain_identity(price_at, context, risk, settings)
        except Exception as error:  # noqa: BLE001 - recorded per state
            rows.append({
                "inception": inception, "model": model, "date": day,
                "spot": float(state.spot), "knocked_in": bool(state.knocked_in),
                "residual": float("nan"), "refinement": float("nan"),
                "status": "error", "error": f"{type(error).__name__}: {error}",
                "banked_residual": float(banked.loc[day, "identity_residual_hands"]),
                "banked_refinement": float(
                    banked.loc[day, "identity_spot_refinement_error_hands"]
                ),
            })
            continue

        residual = samples[-1].residual_hands
        refinement = abs(residual - samples[-2].residual_hands)
        budget = settings.delta_tolerance_hands
        if abs(residual) + refinement <= budget:
            status = "pass"
        elif abs(residual) - refinement > budget:
            status = "fail"
        else:
            status = "inconclusive"

        rows.append({
            "inception": inception, "model": model, "date": day,
            "spot": float(state.spot), "knocked_in": bool(state.knocked_in),
            "residual": residual, "refinement": refinement,
            "status": status, "error": "",
            "banked_residual": float(banked.loc[day, "identity_residual_hands"]),
            "banked_refinement": float(
                banked.loc[day, "identity_spot_refinement_error_hands"]
            ),
        })
    return rows


def _job(payload):
    key, cell_path, stretch = payload
    started = time.perf_counter()
    rows = audit_group(key, cell_path, stretch)
    return key, rows, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--stretch", type=float, default=0.02)
    parser.add_argument("--off", action="store_true",
                        help="run with the option OFF, to validate against banked")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    stretch = None if args.off else args.stretch
    groups = representative_cells()
    payloads = [(key, str(cell), stretch) for key, cell in sorted(groups.items())]
    if args.limit:
        payloads = payloads[: args.limit]

    print(f"{len(payloads)} (inception, model) groups, "
          f"align_cell_stretch={stretch}, workers={args.workers}", flush=True)

    started = time.perf_counter()
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_job, p): p[0] for p in payloads}
        for future in as_completed(futures):
            key, group_rows, elapsed = future.result()
            rows.extend(group_rows)
            done += 1
            print(f"  [{done}/{len(payloads)}] {key[0]} {key[1]:14} "
                  f"{len(group_rows):4d} states {elapsed:7.1f}s "
                  f"(total {time.perf_counter() - started:.0f}s)", flush=True)

    frame = pd.DataFrame(rows).sort_values(["inception", "model", "date"])
    frame.to_csv(args.out, index=False)
    print(f"\nwrote {len(frame)} states to {args.out} "
          f"in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
