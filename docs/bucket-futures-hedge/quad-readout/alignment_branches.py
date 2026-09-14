"""The two alignment branches, priced side by side across the crossover.

align_priority forces the pin to the knock-in or the knock-out barrier.
Each forced branch should be smooth in spot. `auto` picks the nearest
barrier, so it FOLLOWS one branch below the crossover and the other above.
The gap between the branches at the switch point is the discontinuity.
"""
from datetime import date
import json
from pathlib import Path
import sys

REPO = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge")
for _k in [k for k in sys.modules if k == "quantark" or k.startswith("quantark.")]:
    del sys.modules[_k]
sys.path[:0] = [str(REPO), str(REPO / "example/snowball_q_term_structure")]

import numpy as np
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

RUN = REPO / ("example/snowball_q_term_structure/data/bucket_hedge_v2/gate_e"
              "/runs/20230703/term_flat_q__front")
HIST = Path("/Users/fuxinyao/quant-ark/example/mo_volmodels/data/history")
DAY, M = "2023-10-26", 200.0

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
cross = float(np.sqrt(ki * ko))

N = int(config["quad_grid"])
READOUT = config.get("quad_readout", "legacy_linear")


def engine(priority):
    return SnowballQuadEngine(QuadParams(grid_points=N, readout=READOUT,
                                         align_priority=priority))


def pv(eng, s):
    return -eng.price(product, env(s))


e_ki, e_ko, e_auto = engine("ki"), engine("ko"), engine("auto")
probe = pv(e_auto, cross)
h_log = 2.0 * np.log(_m.constant_c) / (N - 1) if False else None

print(f"KI {ki:.2f}   KO {ko:.2f}   crossover {cross:.2f}   spot that day {spot0:.2f}")
print(f"barrier separation log(KO/KI) = {np.log(ko/ki):.6f}")

spots = cross + np.array([-4.0, -3.0, -2.0, -1.0, -0.25, 0.25, 1.0, 2.0, 3.0, 4.0])
print(f"\n{'spot':>9} {'offset':>8} {'pv (pin KI)':>16} {'pv (pin KO)':>16} "
      f"{'pv (auto)':>16} {'auto follows':>13} {'KO-KI':>12}")
recs = []
for s in spots:
    a, b, c = pv(e_ki, s), pv(e_ko, s), pv(e_auto, s)
    follows = "KI" if abs(c - a) < abs(c - b) else "KO"
    recs.append((s, a, b, c))
    print(f"{s:9.2f} {s-cross:8.2f} {a:16.4f} {b:16.4f} {c:16.4f} "
          f"{follows:>13} {b-a:12.4f}")

r = np.array(recs)
print(f"\nbranch gap (pin KO minus pin KI), mean {np.mean(r[:,2]-r[:,1]):.2f}, "
      f"spread {np.ptp(r[:,2]-r[:,1]):.2f}")
for name, col in (("pin KI", 1), ("pin KO", 2), ("auto", 3)):
    d = np.diff(r[:, col]) / np.diff(r[:, 0])
    print(f"{name:8} secant delta across the scan: "
          f"min {d.min():12.1f}  max {d.max():12.1f}  "
          f"range/|mean| {np.ptp(d)/abs(d.mean()):6.3f}")
