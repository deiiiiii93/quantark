"""The 1%-vs-0.25% secant gap, re-measured under the aligned engine.

Two of the five one-contract states in the banked measurement were the
alignment crossover, not secant error. The fix removed that mechanism, so the
question "is a 1% pricing bump good enough against the desk's 0.25% rebalance
band?" has to be re-asked on the fixed engine.

pricing_delta_hedge_gap_hands is computed INSIDE audit_held_book, so the
aligned replay's carry_audit_mode=none leaves it NaN. It does not need the
audit: the gap is

    product_delta_hands  -  direct_frozen_curve_book_delta({}, 0.25% step)/m_ref

and the first term -- the reported 1% pricing delta -- is written to every row
by the recorder regardless of audit mode. Only the 0.25% secant is missing,
which is one central bump: two price calls per state.

Both terms price the product with NO holdings, so the gap is holdings-free and
one hedge policy per (inception, model) covers every distinct market state:
58 cells, 7,558 states, rather than 406 cells and 52,906 rows.

  validate  recompute the banked arm, where the recorder wrote the column
  measure   every live date of the aligned arm
"""
from datetime import date as date_cls
import json
import math
import os
import sys
import time
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
from quantark.backtest.replay.carry_risk import direct_frozen_curve_book_delta
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

BANKED = STUDY / "data/bucket_hedge_v2/gate_e/runs"
ALIGNED = REPO / "output/snowball_q_term_structure/bucket_hedge_v3/runs"
POLICY = "front"          # any single policy: the gap carries no holdings
HEDGE_REL = 0.0025        # the desk's rebalance resolution
OUT = REPO / "docs/bucket-futures-hedge/quad-readout/secant_gap_states.csv.gz"

HISTORY = C.load_history(HIST)
CALENDAR = C.TradingCalendar.from_frames(HISTORY)


class Cell:
    def __init__(self, run_dir: Path):
        self.summary = json.loads((run_dir / "run_summary.json").read_text())
        self.config = json.loads((run_dir / "run_config.json").read_text())
        self.states = pd.read_csv(run_dir / "states.csv").set_index("date")
        self.legs = pd.read_csv(run_dir / "hedge_legs.csv")
        self.attribution = pd.read_csv(
            run_dir / "hedge_attribution.csv").set_index("date")
        self.m_ref = float(self.attribution["reference_multiplier"].dropna().iloc[0])
        self.terms = C.build_terms(
            date_cls.fromisoformat(self.summary["inception"]), CALENDAR)
        self.engine = SnowballQuadEngine(QuadParams(
            grid_points=self.config["quad_grid"],
            readout=self.config["quad_readout"],
            align_cell_stretch=self.config.get("align_cell_stretch")))
        self.original = C.build_product(
            self.terms, self.summary["s0"], self.summary["coupon"])
        self.ko = C.KO_PCT * float(self.summary["s0"])

    def live_dates(self):
        out = []
        for day, rows in self.legs.groupby("date"):
            day = str(day)
            if not rows.is_curve_node.any() or day not in self.states.index:
                continue
            if day not in self.attribution.index:
                continue
            pricing = self.attribution.loc[day, "product_delta_hands"]
            if isinstance(pricing, pd.Series) or not math.isfinite(float(pricing)):
                continue
            out.append(day)
        return sorted(out)

    def gap_hands(self, day):
        """reported 1% pricing delta minus the 0.25% hedge secant, in hands."""
        state = self.states.loc[day]
        rows = self.legs[self.legs.date == day]
        quotes = tuple(
            IndexFuturesQuote(
                contract=r["contract"], maturity=float(r["tenor_years"]),
                price=float(r["price"]), multiplier=float(r["multiplier"]))
            for r in rows[rows.is_curve_node].to_dict("records"))
        context = CarryCurveContext(
            quotes, float(state.spot), FlatRateCurve(rate=float(state.rate)),
            self.config["extrapolation"], "CSI1000", pd.Timestamp(day))

        def env(spot, dividend):
            return PricingEnvironment(
                spot_quote=SpotQuote(spot=spot),
                vol_surface=FlatVolSurface(volatility=float(state.volatility)),
                rate_curve=context.rate_curve, div_yield=dividend,
                valuation_date=pd.Timestamp(day).to_pydatetime())

        tracker = AutocallableLifecycleTracker(
            product=self.original, quantity=-1.0,
            start_date=pd.Timestamp(self.summary["inception"]))
        tracker.lifecycle.knocked_in = bool(state.knocked_in)
        product = tracker.product_for_pricing(
            pd.Timestamp(day), env(context.spot, context.dividend()))

        def price_at(spot, dividend):
            return -self.engine.price(product, env(spot, dividend))

        secant = direct_frozen_curve_book_delta(
            price_at, context, {}, HEDGE_REL * context.spot)
        pricing = float(self.attribution.loc[day, "product_delta_hands"])
        return pricing - secant / self.m_ref


def cells_of(root):
    return sorted((p for p in root.glob(f"*/*__{POLICY}")
                   if (p / "run_summary.json").exists()),
                  key=lambda p: (p.parent.name, p.name))


print("=" * 78, flush=True)
print("validate against the banked arm, where the recorder wrote the column")
print("=" * 78, flush=True)
banked = cells_of(BANKED)
print(f"{len(banked)} banked {POLICY} cells", flush=True)
worst, checked = 0.0, 0
for path in banked[::len(banked) // 4][:4]:
    cell = Cell(path)
    days = cell.live_dates()
    here = 0.0
    for day in days[::max(1, len(days) // 5)][:5]:
        recorded = float(cell.attribution.loc[day, "pricing_delta_hedge_gap_hands"])
        if not math.isfinite(recorded):
            continue
        here = max(here, abs(cell.gap_hands(day) - recorded))
        checked += 1
    worst = max(worst, here)
    print(f"  {path.parent.name}/{path.name:26s} worst |harness - recorded| "
          f"{here:.3e}", flush=True)
print(f"\n{checked} states checked, worst disagreement {worst:.3e} hands", flush=True)
assert worst < 1e-6, "harness does not reproduce the recorder"

print(f"\n{'=' * 78}")
print("measure: every live date of the aligned arm")
print("=" * 78, flush=True)
aligned = cells_of(ALIGNED)
started = time.perf_counter()
records = []
for n, path in enumerate(aligned, 1):
    cell = Cell(path)
    for day in cell.live_dates():
        spot = float(cell.states.loc[day, "spot"])
        records.append({
            "inception": cell.summary["inception"],
            "model": cell.summary["model"],
            "date": day,
            "gap": cell.gap_hands(day),
            "spot": spot,
            "ko": cell.ko,
            "dist_to_ko_pct": 100.0 * (spot / cell.ko - 1.0),
            "knocked_in": bool(cell.states.loc[day, "knocked_in"]),
            "pricing_delta_hands": float(
                cell.attribution.loc[day, "product_delta_hands"]),
        })
    if n % 10 == 0 or n == len(aligned):
        print(f"  [{n}/{len(aligned)}] {len(records)} states, "
              f"{time.perf_counter() - started:.0f}s", flush=True)

df = pd.DataFrame(records)
df["abs_gap"] = df["gap"].abs()
df.to_csv(OUT, index=False, compression="gzip")
print(f"\n{len(df)} states, {time.perf_counter() - started:.0f}s -> {OUT}")

a = df["abs_gap"]
print(f"\naligned arm    mean {a.mean():.4f}  p95 {a.quantile(0.95):.4f}  "
      f"p99 {a.quantile(0.99):.4f}  max {a.max():.4f}")
print(f"               over 0.5 contracts: {int((a > 0.5).sum())}   "
      f"over 1.0: {int((a > 1.0).sum())}")
print("\nbanked arm (recorded) for comparison: mean 0.0430  p95 0.1305  "
      "p99 0.3533  max 2.7103,  over 0.5: 44   over 1.0: 10")

print("\nstates over 1.0 contract")
big = df[df.abs_gap > 1.0].sort_values("abs_gap", ascending=False)
for _, r in big.iterrows():
    print(f"  {r['date']}  {r['model']:14} incep {r['inception']}  "
          f"gap {r['gap']:+7.3f}  spot {r['spot']:8.1f}  "
          f"dist to KO {r['dist_to_ko_pct']:+7.2f}%  KI {r['knocked_in']}")

print("\nabs gap by distance to the knock-out barrier")
print(f"{'band':>16} {'n':>6} {'mean':>9} {'p99':>9} {'max':>9}")
for lo, hi in [(-100, -20), (-20, -10), (-10, -5), (-5, -3), (-3, -2),
               (-2, -1), (-1, 0), (0, 100)]:
    sub = df[(df.dist_to_ko_pct >= lo) & (df.dist_to_ko_pct < hi)]
    if sub.empty:
        continue
    s = sub["abs_gap"]
    print(f"{f'[{lo},{hi})%':>16} {len(sub):6d} {s.mean():9.4f} "
          f"{s.quantile(0.99):9.4f} {s.max():9.4f}")
