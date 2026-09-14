"""What does it cost to leave the OTHER barrier off-node?

Pins the knock-in and the knock-out in turn and refines the lattice. As h
shrinks, the un-pinned barrier's distance to the nearest node shrinks with
it, so both branches must converge to the same price. The spread between
them at the PRODUCTION resolution is the price of the choice; the distance
from the converged value is the price of being off-node.
"""
from datetime import date
import json
from pathlib import Path
import sys
import time

REPO = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge")
for _k in [k for k in sys.modules if k == "quantark" or k.startswith("quantark.")]:
    del sys.modules[_k]
sys.path[:0] = [str(REPO), str(REPO / "example/snowball_q_term_structure")]

import numpy as np
import pandas as pd
import quantark
assert str(REPO) in quantark.__file__, quantark.__file__
import _common as C
from quantark.asset.equity.engine.quad import quad_math as QM
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.asset.equity.param import QuadParams
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

RUN = REPO / ("example/snowball_q_term_structure/data/bucket_hedge_v2/gate_e"
              "/runs/20230703/term_flat_q__front")
HIST = Path("/Users/fuxinyao/quant-ark/example/mo_volmodels/data/history")
DAY = "2023-10-26"

summary = json.loads((RUN / "run_summary.json").read_text())
config = json.loads((RUN / "run_config.json").read_text())
states = pd.read_csv(RUN / "states.csv").set_index("date")
legs = pd.read_csv(RUN / "hedge_legs.csv")
history = C.load_history(HIST)
terms = C.build_terms(date.fromisoformat(summary["inception"]),
                      C.TradingCalendar.from_frames(history))
original = C.build_product(terms, summary["s0"], summary["coupon"])
st = states.loc[DAY]
spot0 = float(st.spot)

rows_q = legs[(legs.date == DAY) & legs.is_curve_node].to_dict("records")
quotes = tuple(IndexFuturesQuote(contract=r["contract"], maturity=float(r["tenor_years"]),
                                 price=float(r["price"]), multiplier=float(r["multiplier"]))
               for r in rows_q)
ctx = CarryCurveContext(quotes, spot0, FlatRateCurve(rate=float(st.rate)),
                        config["extrapolation"], "CSI1000", pd.Timestamp(DAY))
div = ctx.dividend()


def env(s):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=s),
        vol_surface=FlatVolSurface(volatility=float(st.volatility)),
        rate_curve=ctx.rate_curve, div_yield=div,
        valuation_date=pd.Timestamp(DAY).to_pydatetime())


tracker = AutocallableLifecycleTracker(product=original, quantity=-1.,
                                       start_date=pd.Timestamp(summary["inception"]))
tracker.lifecycle.knocked_in = bool(st.knocked_in)
product = tracker.product_for_pricing(pd.Timestamp(DAY), env(spot0))
ki = float(product.barrier_config.ki_barrier)
ko = product.barrier_config.ko_barrier
ko = float(max(ko) if isinstance(ko, list) else ko)
print(f"spot {spot0:.2f}   KI {ki:.2f}   KO {ko:.2f}   "
      f"KI monitoring: {getattr(product.barrier_config, 'ki_monitoring_mode', None)}")

seen = []
orig_init = QM.QuadratureMath.__init__


def spy(self, *a, **k):
    orig_init(self, *a, **k)
    seen.append(self)


QM.QuadratureMath.__init__ = spy


def price(n, priority):
    seen.clear()
    t = time.perf_counter()
    pv = -SnowballQuadEngine(QuadParams(
        grid_points=n, readout=config.get("quad_readout", "legacy_linear"),
        align_priority=priority)).price(product, env(spot0))
    m = seen[0]
    off = (np.log(ko / ki) / m.h) % 1.0
    off = min(off, 1.0 - off)
    return pv, m.grid_x, m.h, off, time.perf_counter() - t


print(f"\n{'N req':>7} {'N used':>7} {'h (pts)':>9} {'off-node (pts)':>15} "
      f"{'pv pin KI':>16} {'pv pin KO':>16} {'KO - KI':>12} {'s':>6}")
res = []
for n in (1225, 2451, 4901, 9801, 19601):
    a, na, h, off, ta = price(n, "ki")
    b, nb, _, _, tb = price(n, "ko")
    res.append((na, h, off, a, b))
    print(f"{n:7d} {na:7d} {h*spot0:9.3f} {off*h*spot0:15.3f} "
          f"{a:16.4f} {b:16.4f} {b-a:12.4f} {ta+tb:6.1f}")

ref = 0.5 * (res[-1][3] + res[-1][4])
print(f"\nreference (finest grid, mean of the two pins): {ref:.4f}")
print(f"{'N used':>7} {'err pin KI':>14} {'err pin KO':>14} "
      f"{'as bp of pv':>13} {'|KO-KI| bp':>12}")
for na, h, off, a, b in res:
    print(f"{na:7d} {a-ref:14.4f} {b-ref:14.4f} "
          f"{max(abs(a-ref), abs(b-ref))/ref*1e4:13.2f} {abs(b-a)/ref*1e4:12.2f}")
