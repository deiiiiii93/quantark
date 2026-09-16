"""The 1%-vs-0.25% secant, drawn at the state where it costs the most.

2025-07-01, inception 2024-12-02, spot 6373.8 against a knock-out barrier at
6503.0 -- 1.99% below it, where the banked fleet's worst gap of 2.71 contracts
sits. Emits the three curves the picture needs:

  price      V(S) on a fine sweep across the barrier, so the kink is visible
  staircase  the local slope of that sweep, which under legacy_linear is a
             staircase whose tread IS the lattice cell
  bump       delta at S0 as a function of the bump width, from sub-cell to 2%,
             with the desk's 0.25% band and the pricing 1% bump marked

Writes secant_delta_profile.json for the write-up. All deltas in hands (one
hand = one IM contract, multiplier 200).
"""
from datetime import date as date_cls
import json
import math
import os
import sys
from pathlib import Path

REPO = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge")
STUDY = REPO / "example/snowball_q_term_structure"
HIST = Path("/Users/fuxinyao/quant-ark/example/mo_volmodels/data/history")
os.environ["PYTHONPATH"] = str(REPO)
for _k in [k for k in list(sys.modules) if k == "quantark" or k.startswith("quantark.")]:
    del sys.modules[_k]
sys.path[:0] = [str(STUDY), str(REPO)]

import pandas as pd
import quantark

assert str(REPO) in quantark.__file__, quantark.__file__

import _common as C
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.asset.equity.param import QuadParams
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

RUN = (REPO / "output/snowball_q_term_structure/bucket_hedge_v3/runs"
       / "20241202/term_flat_q__front")
DAY = "2025-07-01"
OUT = REPO / "docs/bucket-futures-hedge/quad-readout/secant_delta_profile.json"

summary = json.loads((RUN / "run_summary.json").read_text())
config = json.loads((RUN / "run_config.json").read_text())
states = pd.read_csv(RUN / "states.csv").set_index("date")
legs = pd.read_csv(RUN / "hedge_legs.csv")
attribution = pd.read_csv(RUN / "hedge_attribution.csv").set_index("date")
state = states.loc[DAY]
rows = legs[legs.date == DAY]
m_ref = float(attribution.loc[DAY, "reference_multiplier"])

quotes = tuple(
    IndexFuturesQuote(contract=r["contract"], maturity=float(r["tenor_years"]),
                      price=float(r["price"]), multiplier=float(r["multiplier"]))
    for r in rows[rows.is_curve_node].to_dict("records"))
context = CarryCurveContext(
    quotes, float(state.spot), FlatRateCurve(rate=float(state.rate)),
    config["extrapolation"], "CSI1000", pd.Timestamp(DAY))
dividend = context.dividend()
S0 = float(context.spot)
KO = C.KO_PCT * float(summary["s0"])

history = C.load_history(HIST)
terms = C.build_terms(date_cls.fromisoformat(summary["inception"]),
                      C.TradingCalendar.from_frames(history))
engine = SnowballQuadEngine(QuadParams(
    grid_points=config["quad_grid"], readout=config["quad_readout"],
    align_cell_stretch=config.get("align_cell_stretch")))
original = C.build_product(terms, summary["s0"], summary["coupon"])


def env(spot):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=float(state.volatility)),
        rate_curve=context.rate_curve, div_yield=dividend,
        valuation_date=pd.Timestamp(DAY).to_pydatetime())


tracker = AutocallableLifecycleTracker(
    product=original, quantity=-1.0, start_date=pd.Timestamp(summary["inception"]))
tracker.lifecycle.knocked_in = bool(state.knocked_in)
product = tracker.product_for_pricing(pd.Timestamp(DAY), env(S0))


def price(spot):
    """The short book's value, the sign the desk carries."""
    return -engine.price(product, env(spot))


def delta_hands(spot, step):
    return (price(spot + step) - price(spot - step)) / (2.0 * step) / m_ref


# --- the price sweep, fine enough to resolve the lattice ------------------
lo, hi, n = S0 * 0.965, S0 * 1.025, 241
sweep = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
prices = [price(s) for s in sweep]
notional = float(config["risk"]["reference_notional"])
print(f"state {DAY}  inception {summary['inception']}  {config['quad_readout']}"
      f"  align={config.get('align_cell_stretch')}")
print(f"spot {S0:.1f}   KO {KO:.1f}   {100 * (S0 / KO - 1):+.2f}%   "
      f"vol {float(state.volatility):.3f}   KI {bool(state.knocked_in)}")

# the local slope of that sweep: a staircase under legacy_linear
step_local = []
for i in range(1, n - 1):
    slope = (prices[i + 1] - prices[i - 1]) / (sweep[i + 1] - sweep[i - 1]) / m_ref
    step_local.append({"spot": sweep[i], "delta": slope})

# --- delta at S0 as a function of bump width ------------------------------
widths, bump = [], []
for i in range(60):
    rel = 0.0004 * math.exp(i * math.log(2.0 / 0.04) / 59)   # 0.04% -> 2%
    widths.append(rel)
for rel in widths:
    bump.append({"rel": rel, "points": rel * S0, "delta": delta_hands(S0, rel * S0)})

d_hedge = delta_hands(S0, 0.0025 * S0)
d_price = delta_hands(S0, 0.01 * S0)
recorded_pricing = float(attribution.loc[DAY, "product_delta_hands"])
print(f"\n0.25% secant  {d_hedge:+.4f} hands   ({0.0025 * S0:.1f} index points)")
print(f"1%    secant  {d_price:+.4f} hands   ({0.01 * S0:.1f} index points)")
print(f"recorded pricing delta {recorded_pricing:+.4f} hands")
print(f"gap (1% - 0.25%)  {recorded_pricing - d_hedge:+.4f} contracts")

# The cell shows itself in the BUMP profile, not in the sweep. While the whole
# central difference stays inside one cell a piecewise-linear readout returns
# the same chord, so delta(h) is exactly flat; the widest such h is half a
# cell. Reading it off slope changes instead picks up the per-spot re-solve
# drift, which moves at every step and says nothing about the cell.
flat = bump[0]["delta"]
plateau = max((r["points"] for r in bump if abs(r["delta"] - flat) < 1e-9),
              default=float("nan"))
tread = 2.0 * plateau
print(f"\ndelta is EXACTLY {flat:.4f} hands for every bump out to "
      f"{plateau:.1f} points, so the lattice cell is at least "
      f"{tread:.1f} index points ({100 * tread / S0:.2f}% of spot)")
print(f"  the desk's 0.25% band is {0.0025 * S0:.1f} points -- "
      f"{'INSIDE that cell' if 0.0025 * S0 < tread else 'wider than the cell'}")
print(f"  the 1% pricing bump is  {0.01 * S0:.1f} points")

OUT.write_text(json.dumps({
    "state": {
        "date": DAY, "inception": summary["inception"],
        "model": "term_flat_q", "spot": S0, "ko": KO,
        "dist_to_ko_pct": 100 * (S0 / KO - 1), "vol": float(state.volatility),
        "knocked_in": bool(state.knocked_in), "notional": notional,
        "readout": config["quad_readout"],
        "align_cell_stretch": config.get("align_cell_stretch"),
        "grid_points": config["quad_grid"], "m_ref": m_ref,
    },
    "price_sweep": [{"spot": s, "price": p} for s, p in zip(sweep, prices)],
    "local_slope": step_local,
    "bump_profile": bump,
    "readings": {
        "hedge_secant_0_25pct": d_hedge,
        "pricing_secant_1pct": d_price,
        "recorded_pricing_delta": recorded_pricing,
        "gap_contracts": recorded_pricing - d_hedge,
        "tread_points": tread,
    },
}, indent=1))
print(f"\nwrote {OUT}")
