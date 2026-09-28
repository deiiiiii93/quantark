"""Cost of using a 1% pricing bump when the desk hedges at 0.25%.

The reference is NOT the tangent. If the book is re-hedged after a 0.25%
move, spot wanders inside that band between rebalances, so the slope that
governs P&L over the holding interval is the secant across it. The 1% bump
is then an approximation to the 0.25% secant, and this measures the error
of that approximation on every live date, under both readouts.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[0]
REPO = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge")
sys.path[:0] = [str(REPO), str(REPO / "example/snowball_q_term_structure")]
import pandas as pd
import _common as C
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.asset.equity.param import QuadParams
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.backtest.replay.carry_risk import direct_frozen_curve_book_delta
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

RUN = REPO / ("example/snowball_q_term_structure/data/bucket_hedge_v2/subset_matched"
              "/runs/20230504/term_flat_q__front")
HIST = Path("/Users/fuxinyao/quant-ark/example/mo_volmodels/data/history")
M = 200.0
HEDGE_BUMP = 0.0025     # the desk's rebalance resolution
PRICING_BUMP = 0.01     # the engine default

summary = json.loads((RUN / "run_summary.json").read_text())
config = json.loads((RUN / "run_config.json").read_text())
states = pd.read_csv(RUN / "states.csv").set_index("date")
legs = pd.read_csv(RUN / "hedge_legs.csv")
grouped = {d: f[f.is_curve_node].to_dict("records") for d, f in legs.groupby("date")}
history = C.load_history(HIST)
terms = C.build_terms(date.fromisoformat(summary["inception"]),
                      C.TradingCalendar.from_frames(history))
original = C.build_product(terms, summary["s0"], summary["coupon"])


def measure(day):
    st = states.loc[day]
    rows = grouped[day]
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
    out = {"date": day, "spot": float(st.spot), "knocked_in": bool(st.knocked_in)}
    for readout in ("legacy_linear", "transition"):
        engine = SnowballQuadEngine(QuadParams(grid_points=config["quad_grid"], readout=readout))

        def price_at(s, q):
            return -engine.price(product, env(s, q))

        d1 = direct_frozen_curve_book_delta(price_at, ctx, {}, PRICING_BUMP * ctx.spot) / M
        d2 = direct_frozen_curve_book_delta(price_at, ctx, {}, HEDGE_BUMP * ctx.spot) / M
        out[f"{readout}_1pct"] = d1
        out[f"{readout}_hedge"] = d2
        out[f"{readout}_gap"] = d1 - d2
    return out


dates = [d for d in states.index if bool(states.loc[d, "alive"]) and not bool(states.loc[d, "matured"])]
rows = []
with ThreadPoolExecutor(max_workers=5) as pool:
    for i, r in enumerate(pool.map(measure, dates), 1):
        rows.append(r)
        if i % 60 == 0:
            print(f"{i}/{len(dates)}", flush=True)
f = pd.DataFrame(rows)
f.to_csv(ROOT / "secant_cost.csv", index=False)

print(f"\n{len(f)} live dates. Error of the 1% secant against the 0.25% hedge secant,\n"
      f"in hands, where 1 hand = 1 IM contract:\n")
print(f"{'readout':16} {'mean':>10} {'p95':>10} {'max':>10} {'>0.5 ctr':>9} {'>1 ctr':>7}")
for r in ("legacy_linear", "transition"):
    g = f[f"{r}_gap"].abs()
    print(f"{r:16} {g.mean():10.4f} {g.quantile(.95):10.4f} {g.max():10.4f} "
          f"{int((g > .5).sum()):9d} {int((g > 1).sum()):7d}")

pre = f[~f.knocked_in]
print(f"\nbefore knock-in only ({len(pre)} dates), where the barrier is live:")
for r in ("legacy_linear", "transition"):
    g = pre[f"{r}_gap"].abs()
    print(f"  {r:16} mean {g.mean():.4f}  max {g.max():.4f}")

worst = f.loc[f["transition_gap"].abs().idxmax()]
print(f"\nworst under transition: {worst['date']} spot {worst['spot']:.0f}  "
      f"1% {worst['transition_1pct']:.4f}  0.25% {worst['transition_hedge']:.4f}  "
      f"gap {worst['transition_gap']:.4f} hands")
