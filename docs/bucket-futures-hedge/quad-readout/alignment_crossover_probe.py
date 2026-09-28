"""Which TERM of the identity spikes on 2023-10-26?

R = (D_frozen - D_pinned - sum_i (F_i/S) B_i) / m_ref

Three terms, three different scenario families. Reachability is ruled out
for the frozen direction. Decompose before hypothesising again.
"""
from datetime import date
import json
from pathlib import Path
import sys

REPO = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge")
sys.path[:0] = [str(REPO), str(REPO / "example/snowball_q_term_structure")]
import pandas as pd
import _common as C
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.asset.equity.param import QuadParams
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.backtest.replay.carry_risk import (
    buckets_of, direct_frozen_curve_book_delta, direct_pinned_delta, sample_buckets)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

RUN = REPO / ("example/snowball_q_term_structure/data/bucket_hedge_v2/gate_e"
              "/runs/20230703/term_flat_q__front")
HIST = Path("/Users/fuxinyao/quant-ark/example/mo_volmodels/data/history")
M, BUMP = 200.0, 0.00025

summary = json.loads((RUN / "run_summary.json").read_text())
config = json.loads((RUN / "run_config.json").read_text())
states = pd.read_csv(RUN / "states.csv").set_index("date")
legs = pd.read_csv(RUN / "hedge_legs.csv")
history = C.load_history(HIST)
terms = C.build_terms(date.fromisoformat(summary["inception"]),
                      C.TradingCalendar.from_frames(history))
original = C.build_product(terms, summary["s0"], summary["coupon"])

print(f"{'date':12} {'spot':>9} {'D_frozen':>14} {'D_pinned':>14} "
      f"{'buckets':>14} {'R hands':>12}")
for day in ("2023-10-24", "2023-10-25", "2023-10-26", "2023-10-27", "2023-10-30"):
    st = states.loc[day]
    rows = legs[(legs.date == day) & legs.is_curve_node].to_dict("records")
    qs = tuple(IndexFuturesQuote(contract=r["contract"], maturity=float(r["tenor_years"]),
                                 price=float(r["price"]), multiplier=float(r["multiplier"]))
               for r in rows)
    ctx = CarryCurveContext(qs, float(st.spot), FlatRateCurve(rate=float(st.rate)),
                            config["extrapolation"], "CSI1000", pd.Timestamp(day))

    def env(s, q):
        return PricingEnvironment(
            spot_quote=SpotQuote(spot=s), vol_surface=FlatVolSurface(volatility=float(st.volatility)),
            rate_curve=ctx.rate_curve, div_yield=q,
            valuation_date=pd.Timestamp(day).to_pydatetime())

    tracker = AutocallableLifecycleTracker(product=original, quantity=-1.,
                                           start_date=pd.Timestamp(summary["inception"]))
    tracker.lifecycle.knocked_in = bool(st.knocked_in)
    product = tracker.product_for_pricing(pd.Timestamp(day), env(ctx.spot, ctx.dividend()))
    engine = SnowballQuadEngine(QuadParams(grid_points=config["quad_grid"],
                                           readout=config.get("quad_readout", "legacy_linear")))

    def price_at(s, q):
        return -engine.price(product, env(s, q))

    step = BUMP * ctx.spot
    dfz = direct_frozen_curve_book_delta(price_at, ctx, {}, step) / M
    dpn = direct_pinned_delta(price_at, ctx, step) / M
    bsum = sum(b.price / ctx.spot * b.bucket_currency
               for b in buckets_of(ctx, sample_buckets(price_at, ctx, 1.))) / M
    print(f"{day:12} {float(st.spot):9.2f} {dfz:14.6f} {dpn:14.6f} "
          f"{bsum:14.6f} {dfz - dpn - bsum:12.6f}")
