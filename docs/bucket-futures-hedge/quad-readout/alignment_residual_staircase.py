"""Did the snap AMPLIFY the delta staircase, or only move where its treads fall?

The eight-point scan is too short to tell: one tread is ~19 index points, so
a short scan either straddles a riser or does not, by luck of where the
lattice sits. Scan a full two treads instead and compare the peak-to-peak
delta with the snap on and off. Equal amplitude => relocation. Larger
amplitude with the snap on => the snap made the readout worse, which would
be a finding against it.
"""
from datetime import date
import json
from pathlib import Path
import sys

REPO = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge")
for _k in [k for k in list(sys.modules) if k == "quantark" or k.startswith("quantark.")]:
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
DAY, STRETCH = "2023-10-26", 0.02

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
ko = float(product.barrier_config.ko_barrier)
cross = float(np.sqrt(ki * ko))
N = int(config["quad_grid"])
READOUT = config.get("quad_readout", "legacy_linear")


def pv(spot, stretch):
    kw = {} if stretch is None else {"align_cell_stretch": stretch}
    return -SnowballQuadEngine(QuadParams(
        grid_points=N, readout=READOUT, align_priority="auto", **kw
    )).price(product, env(spot))


# two treads wide, sampled finely enough to see a riser
spots = cross + np.linspace(-20.0, 20.0, 41)
print(f"scan {spots[0]:.1f} to {spots[-1]:.1f} ({spots[-1]-spots[0]:.0f} pts, "
      f"one tread is about 18.8 pts), {len(spots)} points, crossover {cross:.2f}")
print(f"\n{'config':12} {'delta min':>12} {'delta max':>12} {'peak-to-peak':>13} "
      f"{'p2p / |mean|':>13} {'in hands':>9}")
out = {}
for stretch, label in ((None, "snap off"), (STRETCH, "snap on")):
    prices = np.array([pv(s, stretch) for s in spots])
    d = np.diff(prices) / np.diff(spots)
    out[label] = d
    print(f"{label:12} {d.min():12.1f} {d.max():12.1f} {np.ptp(d):13.1f} "
          f"{np.ptp(d)/abs(d.mean()):13.4f} {np.ptp(d)/200.0:9.3f}")

mid = 0.5 * (spots[1:] + spots[:-1])
print(f"\n{'spot':>9} {'offset':>8} {'delta off':>12} {'delta on':>12}")
for s, a, b in zip(mid, out["snap off"], out["snap on"]):
    mark = "  <- crossover" if abs(s - cross) < 1.0 else ""
    print(f"{s:9.2f} {s-cross:8.2f} {a:12.1f} {b:12.1f}{mark}")
