"""Does the whole-cell fix clear the carry-audit breach on 2023-10-26?

The identity is exact in theory:

    D_frozen = D_pinned + sum_i (F_i / S) * B_i

so the residual R, in hands, is pure numerical error. It breached the
0.01-hand budget on this one date and nowhere else. Recompute it with
align_cell_stretch on and off, on the failing date and its neighbours.

This is the failing state only, not the fleet: a full re-run is the gate,
this is the mechanism.
"""
from datetime import date
import json
from pathlib import Path
import sys

REPO = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge")
for _k in [k for k in list(sys.modules) if k == "quantark" or k.startswith("quantark.")]:
    del sys.modules[_k]
sys.path[:0] = [str(REPO), str(REPO / "example/snowball_q_term_structure")]

import pandas as pd
import quantark
assert str(REPO) in quantark.__file__, quantark.__file__
import _common as C
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.asset.equity.param import QuadParams
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.backtest.replay.carry_risk import (
    buckets_of,
    direct_frozen_curve_book_delta,
    direct_pinned_delta,
    sample_buckets,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

RUN = REPO / ("example/snowball_q_term_structure/data/bucket_hedge_v2/gate_e"
              "/runs/20230703/term_flat_q__front")
HIST = Path("/Users/fuxinyao/quant-ark/example/mo_volmodels/data/history")
M, BUMP, STRETCH, BUDGET = 200.0, 0.00025, 0.02, 0.01

summary = json.loads((RUN / "run_summary.json").read_text())
config = json.loads((RUN / "run_config.json").read_text())
states = pd.read_csv(RUN / "states.csv").set_index("date")
legs = pd.read_csv(RUN / "hedge_legs.csv")
history = C.load_history(HIST)
terms = C.build_terms(date.fromisoformat(summary["inception"]),
                      C.TradingCalendar.from_frames(history))
original = C.build_product(terms, summary["s0"], summary["coupon"])

DAYS = ("2023-10-24", "2023-10-25", "2023-10-26", "2023-10-27", "2023-10-30")


def residual(day, stretch):
    st = states.loc[day]
    rows = legs[(legs.date == day) & legs.is_curve_node].to_dict("records")
    quotes = tuple(
        IndexFuturesQuote(contract=r["contract"], maturity=float(r["tenor_years"]),
                          price=float(r["price"]), multiplier=float(r["multiplier"]))
        for r in rows
    )
    ctx = CarryCurveContext(quotes, float(st.spot), FlatRateCurve(rate=float(st.rate)),
                            config["extrapolation"], "CSI1000", pd.Timestamp(day))

    def env(s, q):
        return PricingEnvironment(
            spot_quote=SpotQuote(spot=s),
            vol_surface=FlatVolSurface(volatility=float(st.volatility)),
            rate_curve=ctx.rate_curve, div_yield=q,
            valuation_date=pd.Timestamp(day).to_pydatetime())

    tracker = AutocallableLifecycleTracker(
        product=original, quantity=-1.0,
        start_date=pd.Timestamp(summary["inception"]))
    tracker.lifecycle.knocked_in = bool(st.knocked_in)
    product = tracker.product_for_pricing(pd.Timestamp(day), env(ctx.spot, ctx.dividend()))

    kwargs = {"grid_points": config["quad_grid"],
              "readout": config.get("quad_readout", "legacy_linear")}
    if stretch is not None:
        kwargs["align_cell_stretch"] = stretch
    engine = SnowballQuadEngine(QuadParams(**kwargs))

    def price_at(s, q):
        return -engine.price(product, env(s, q))

    step = BUMP * ctx.spot
    frozen = direct_frozen_curve_book_delta(price_at, ctx, {}, step) / M
    pinned = direct_pinned_delta(price_at, ctx, step) / M
    buckets = sum(
        b.price / ctx.spot * b.bucket_currency
        for b in buckets_of(ctx, sample_buckets(price_at, ctx, 1.0))
    ) / M
    return float(st.spot), frozen - pinned - buckets


print(f"identity residual R in hands, budget {BUDGET}\n")
print(f"{'date':12} {'spot':>9} {'R off':>12} {'R on':>12} {'breach off':>11} "
      f"{'breach on':>10}")
for day in DAYS:
    spot, r_off = residual(day, None)
    _, r_on = residual(day, STRETCH)
    print(f"{day:12} {spot:9.2f} {r_off:12.6f} {r_on:12.6f} "
          f"{str(abs(r_off) > BUDGET):>11} {str(abs(r_on) > BUDGET):>10}")
