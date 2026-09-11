"""Run the production audit on archived live states without replaying hedges.

PV agreement is checked before reusing recorded Greeks and actual holdings.
Original artifacts are read only; the requested output CSV is a new audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "example/snowball_q_term_structure")]
import pandas as pd
import _common as C
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.asset.equity.param import QuadParams
from quantark.backtest.futures_risk import CarryRiskSettings, FuturesBookRisk, FuturesBucket
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.backtest.replay.carry_risk import audit_held_book
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--history-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    summary = json.loads((args.run_dir / "run_summary.json").read_text())
    config = json.loads((args.run_dir / "run_config.json").read_text())
    states = pd.read_csv(args.run_dir / "states.csv").set_index("date")
    attrs = pd.read_csv(args.run_dir / "hedge_attribution.csv").set_index("date")
    legs = pd.read_csv(args.run_dir / "hedge_legs.csv")
    grouped = {day: frame[frame.is_curve_node].to_dict("records") for day, frame in legs.groupby("date")}
    history = C.load_history(args.history_dir)
    terms = C.build_terms(date.fromisoformat(summary["inception"]), C.TradingCalendar.from_frames(history))
    original = C.build_product(terms, summary["s0"], summary["coupon"])
    settings = CarryRiskSettings(reference_notional=summary["notional"], audit_spot_bump_rel=.01)

    def measure(day):
        st = states.loc[day]
        rows = grouped[day]
        qs = tuple(IndexFuturesQuote(contract=r["contract"], maturity=float(r["tenor_years"]),
                                    price=float(r["price"]), multiplier=float(r["multiplier"])) for r in rows)
        context = CarryCurveContext(qs, float(st.spot), FlatRateCurve(rate=float(st.rate)),
                                    config["extrapolation"], "CSI1000", pd.Timestamp(day))

        def env(s, q):
            return PricingEnvironment(spot_quote=SpotQuote(spot=s),
                vol_surface=FlatVolSurface(volatility=float(st.volatility)),
                rate_curve=context.rate_curve, div_yield=q, valuation_date=pd.Timestamp(day).to_pydatetime())

        tracker = AutocallableLifecycleTracker(product=original, quantity=-1.,
                                               start_date=pd.Timestamp(summary["inception"]))
        tracker.lifecycle.knocked_in = bool(st.knocked_in)
        product = tracker.product_for_pricing(pd.Timestamp(day), env(context.spot, context.dividend()))
        engine = SnowballQuadEngine(QuadParams(grid_points=config["quad_grid"],
                                               readout=config.get("quad_readout", "legacy_linear")))

        def price_at(s, q):
            return -engine.price(product, env(s, q))

        base = price_at(context.spot, context.dividend())
        if abs(base - float(st.product_mtm)) > .01:
            raise ValueError(f"{day}: archived PV mismatch {base - st.product_mtm}")
        buckets = tuple(FuturesBucket(r["contract"], None, float(r["tenor_years"]),
                        float(r["price"]), float(r["multiplier"]), float(r["bucket_currency"])) for r in rows)
        risk = FuturesBookRisk(context.spot, float(attrs.loc[day, "product_delta_hands"]) * 200., buckets)
        result = audit_held_book(price_at, context, risk,
                                 {r["contract"]: float(r["held_after"]) for r in rows}, settings=settings)
        return dict(date=day, old_status=attrs.loc[day, "audit_status"], status=result.status,
            reason=result.reason, identity_status=result.identity_status,
            identity_residual_hands=result.identity_residual_hands,
            identity_spot_refinement_error_hands=result.identity_spot_refinement_error_hands,
            finite_bump_identity_residual_hands=result.finite_bump_identity_residual_hands,
            finite_bump_identity_status=result.finite_bump_identity_status,
            finite_bump_identity_reason=result.finite_bump_identity_reason,
            pricing_delta_local_gap_hands=result.pricing_delta_local_gap_hands,
            net_delta_error_hands=result.net_delta_audit_error_hands,
            worst_rhoq_error_bp=result.worst_rhoq_error_bp, price_calls=result.price_calls,
            base_pv_error_cny=base-float(st.product_mtm),
            identity_ladder=json.dumps([asdict(s) for s in result.identity_samples]))

    dates = [day for day in attrs.index if bool(states.loc[day, "alive"]) and not bool(states.loc[day, "matured"])]
    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(measure, dates):
            out.append(row)
            if len(out) % 30 == 0:
                print(f"{len(out)}/{len(dates)}", flush=True)
    frame = pd.DataFrame(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(json.dumps(dict(count=len(frame), statuses=frame.status.value_counts().to_dict(),
        max_identity_hands=float(frame.identity_residual_hands.abs().max()),
        max_refinement_hands=float(frame.identity_spot_refinement_error_hands.max()),
        max_pv_error_cny=float(frame.base_pv_error_cny.abs().max()),
        finite_bump_statuses=frame.finite_bump_identity_status.value_counts().to_dict())), flush=True)


if __name__ == "__main__":
    main()
