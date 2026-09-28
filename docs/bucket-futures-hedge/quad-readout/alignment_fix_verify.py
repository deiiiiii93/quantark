"""Follow-ups the first pass could not settle.

  a. how far apart are the pin-KI and pin-KO lattices once snapped? (the
     first check compared against a recomputed linspace, which answers a
     different question -- float addition is not associative)
  b. with the snap on, is `auto` identical to a forced pin?
  c. what is left of the 2.7% delta variation -- alignment, or the readout
     staircase that is already on the record as a separate defect?
  d. how much does the snap move the price? that is the golden rebase.
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

seen = []
orig_init = QM.QuadratureMath.__init__


def spy(self, *a, **k):
    orig_init(self, *a, **k)
    seen.append(self)


QM.QuadratureMath.__init__ = spy
N = int(config["quad_grid"])
READOUT = config.get("quad_readout", "legacy_linear")


def run(spot, priority, stretch):
    seen.clear()
    kw = {} if stretch is None else {"align_cell_stretch": stretch}
    pv = -SnowballQuadEngine(QuadParams(
        grid_points=N, readout=READOUT, align_priority=priority, **kw
    )).price(product, env(spot))
    return pv, seen[0]


print("=== a. distance between the two pinned lattices ===")
for stretch, label in ((None, "OFF"), (STRETCH, "ON")):
    pv_ki, m_ki = run(spot0, "ki", stretch)
    pv_ko, m_ko = run(spot0, "ko", stretch)
    gap = np.abs(m_ki.grid - m_ko.grid).max()
    print(f"  {label:4} max |grid_ki - grid_ko| {gap:.3e} log = {gap*spot0:.3e} pts"
          f"   pv difference {pv_ko - pv_ki:.6e}")

print("\n=== b. with the snap on, is auto identical to a forced pin? ===")
pv_auto, m_auto = run(spot0, "auto", STRETCH)
pv_ki, m_ki = run(spot0, "ki", STRETCH)
print(f"  max |grid_auto - grid_ki| {np.abs(m_auto.grid-m_ki.grid).max():.3e}"
      f"   pv difference {pv_auto - pv_ki:.6e}")

print("\n=== c. what is the residual delta variation made of? ===")
spots = cross + np.array([-3.0, -2.0, -1.0, -0.25, 0.25, 1.0, 2.0, 3.0])
for stretch, priority, label in ((STRETCH, "auto", "snap on, auto"),
                                 (STRETCH, "ki", "snap on, pin KI"),
                                 (None, "ki", "snap off, pin KI")):
    pvs = np.array([run(s, priority, stretch)[0] for s in spots])
    d = np.diff(pvs) / np.diff(spots)
    print(f"  {label:16} delta min {d.min():11.1f} max {d.max():11.1f} "
          f"range/|mean| {np.ptp(d)/abs(d.mean()):7.4f}")

print("\n=== d. price move from turning the snap on (the golden rebase) ===")
off, _ = run(spot0, "auto", None)
on, m_on = run(spot0, "auto", STRETCH)
print(f"  off {off:.4f}   on {on:.4f}   move {on-off:+.4f} "
      f"= {(on-off)/off*1e4:+.2f} bp   h/h0 {m_on.cell_snap_ratio:.6f}")
print(f"  grid_x off/on: {_.grid_x} / {m_on.grid_x}   (must be equal)")
