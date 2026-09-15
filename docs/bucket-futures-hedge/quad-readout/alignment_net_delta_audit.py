"""net_delta_audit_error under the aligned engine.

The Gate E replay ran with carry_audit_mode=none, so this column -- the last
gate left unmeasured after the alignment fix -- is NaN in every aligned row.
It is holdings-dependent, so unlike the chain identity it cannot be collapsed
across hedge policies: every (inception, cell, date) is its own state, 98,252
of them, and a full audit replay costs 16 hours.

It does not need the full audit. The error is

    direct_frozen_curve_book_delta(holdings)/m_ref  -  held_book_risk(risk, holdings)/m_ref

and the recorder already writes the second term to EVERY row as
``net_delta_hands`` -- it comes from the risk the hedger used, not from the
audit, so it survives carry_audit_mode=none. Only the direct reprice is
missing, which is one central bump: two price calls per state.

That is also exactly what the audit compares. The mapped side is the
recorder's own algebraic prediction; the direct side is an independent
revaluation. Reconstructing the mapped side here instead would test my
reconstruction, not the recorder's.

Two phases:

  validate  recompute the banked arm, where the recorder DID write the error
            column, and check this harness reproduces it
  measure   run the aligned arm on a stratified sample of every cell

Usage: python alignment_net_delta_audit.py [per-cell sample size]
"""
from datetime import date as date_cls
import json
import math
import os
import statistics
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
from quantark.backtest.futures_risk import CarryRiskSettings
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.backtest.replay.carry_risk import direct_frozen_curve_book_delta
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

BANKED = STUDY / "data/bucket_hedge_v2/gate_e"
ALIGNED = REPO / "output/snowball_q_term_structure/bucket_hedge_v3"
PER_CELL = int(sys.argv[1]) if len(sys.argv) > 1 else 8

# The trading calendar is the same for every cell; loading it per cell turned
# out to cost more than the pricing did.
HISTORY = C.load_history(HIST)
CALENDAR = C.TradingCalendar.from_frames(HISTORY)


class Cell:
    """One run directory, loaded once and measured many times."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.summary = json.loads((run_dir / "run_summary.json").read_text())
        self.config = json.loads((run_dir / "run_config.json").read_text())
        self.states = pd.read_csv(run_dir / "states.csv").set_index("date")
        self.legs = pd.read_csv(run_dir / "hedge_legs.csv")
        self.attribution = pd.read_csv(
            run_dir / "hedge_attribution.csv").set_index("date")
        risk_cfg = self.config["risk"]
        self.settings = CarryRiskSettings(
            reference_notional=float(risk_cfg["reference_notional"]),
            delta_tolerance_hands=float(risk_cfg["delta_tolerance_hands"]),
            rhoq_tolerance_bp=float(risk_cfg["rhoq_tolerance_bp"]),
            futures_bump_points=float(risk_cfg["futures_bump_points"]),
            audit_yield_bump=float(risk_cfg["audit_yield_bump"]),
            identity_spot_bumps_rel=tuple(risk_cfg["identity_spot_bumps_rel"]),
            hedge_resolution_rel=risk_cfg.get("hedge_resolution_rel"),
        ).resolved(
            reference_notional=float(risk_cfg["reference_notional"]),
            audit_spot_bump_rel=risk_cfg.get("audit_spot_bump_rel") or 0.01,
        )
        self.terms = C.build_terms(
            date_cls.fromisoformat(self.summary["inception"]), CALENDAR)
        self.engine = SnowballQuadEngine(QuadParams(
            grid_points=self.config["quad_grid"],
            readout=self.config["quad_readout"],
            align_cell_stretch=self.config.get("align_cell_stretch")))
        self.original = C.build_product(
            self.terms, self.summary["s0"], self.summary["coupon"])
        self.m_ref = float(self.settings.reference_multiplier)

    def measurable_dates(self):
        """Dates with curve nodes, a market state and a recorded mapped delta."""
        out = []
        for day, rows in self.legs.groupby("date"):
            day = str(day)
            if not rows.is_curve_node.any() or day not in self.states.index:
                continue
            if day not in self.attribution.index:
                continue
            mapped = self.attribution.loc[day, "net_delta_hands"]
            if isinstance(mapped, pd.Series) or not math.isfinite(float(mapped)):
                continue
            out.append(day)
        return sorted(out)

    def error_hands(self, day):
        """direct - mapped, in hands, for the holdings actually carried."""
        state = self.states.loc[day]
        rows = self.legs[self.legs.date == day]
        quotes = tuple(
            IndexFuturesQuote(
                contract=r["contract"], maturity=float(r["tenor_years"]),
                price=float(r["price"]), multiplier=float(r["multiplier"]))
            for r in rows[rows.is_curve_node].to_dict("records")
        )
        holdings = {
            r["contract"]: float(r["held_after"])
            for r in rows.to_dict("records")
            if float(r["held_after"]) != 0.0
        }
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

        step = float(self.settings.audit_spot_bump_rel) * context.spot
        direct = direct_frozen_curve_book_delta(price_at, context, holdings, step)
        mapped = float(self.attribution.loc[day, "net_delta_hands"])
        return direct / self.m_ref - mapped

    def recorded_error(self, day):
        value = self.attribution.loc[day, "net_delta_audit_error_hands"]
        return float(value)


def cells_of(root: Path):
    return sorted(
        (p for p in root.glob("runs/*/*") if (p / "run_summary.json").exists()),
        key=lambda p: (p.parent.name, p.name))


def sample_dates(cell: Cell, k: int):
    """First, last, the largest residual, the largest book, then evenly spaced."""
    days = cell.measurable_dates()
    if not days:
        return []
    picked = {days[0], days[-1]}
    attr = cell.attribution.reindex(days)
    for column in ("net_delta_hands", "gross_contracts"):
        if column in attr and attr[column].notna().any():
            picked.add(str(attr[column].abs().idxmax()))
    stride = max(1, len(days) // max(1, k - len(picked)))
    picked.update(days[::stride])
    return sorted(picked)


print("=" * 78, flush=True)
print("validate: recompute the banked arm, where the recorder wrote the error")
print("=" * 78, flush=True)
banked = cells_of(BANKED)
print(f"{len(banked)} banked cells; 6 states in each of 5, spread across the fleet",
      flush=True)
worst_overall, checked = 0.0, 0
for path in banked[::len(banked) // 5][:5]:
    cell = Cell(path)
    worst_here = 0.0
    for day in sample_dates(cell, 6):
        recorded = cell.recorded_error(day)
        if not math.isfinite(recorded):
            continue
        worst_here = max(worst_here, abs(cell.error_hands(day) - recorded))
        checked += 1
    worst_overall = max(worst_overall, worst_here)
    print(f"  {path.parent.name}/{path.name:36s} worst |harness - recorded| "
          f"{worst_here:.3e}", flush=True)
print(f"\n{checked} states checked, worst disagreement {worst_overall:.3e} hands",
      flush=True)
assert worst_overall < 1e-9, "harness does not reproduce the recorder"

print(f"\n{'=' * 78}")
print(f"measure: the aligned arm, up to {PER_CELL} states per cell")
print("=" * 78, flush=True)
aligned = cells_of(ALIGNED)
started = time.perf_counter()
errors, worst = [], None
for n, path in enumerate(aligned, 1):
    cell = Cell(path)
    for day in sample_dates(cell, PER_CELL):
        value = cell.error_hands(day)
        errors.append(abs(value))
        if worst is None or abs(value) > abs(worst[0]):
            worst = (value, path.parent.name, path.name, day)
    if n % 50 == 0 or n == len(aligned):
        print(f"  [{n}/{len(aligned)}] {len(errors)} states, "
              f"max |error| {max(errors):.3e}, {time.perf_counter() - started:.0f}s",
              flush=True)

errors.sort()
print(f"\n{len(errors)} states over {len(aligned)} cells, "
      f"{time.perf_counter() - started:.0f}s")
print(f"  max     {errors[-1]:.3e} hands")
print(f"  p99     {errors[int(0.99 * (len(errors) - 1))]:.3e}")
print(f"  p95     {errors[int(0.95 * (len(errors) - 1))]:.3e}")
print(f"  median  {statistics.median(errors):.3e}")
print(f"  worst state: {worst[0]:+.3e} hands  {worst[1]} {worst[2]} {worst[3]}")
print("\nbanked Gate E reported net_delta_audit_error_max 3.819877747446299e-12")
print("tolerance (delta_tolerance_hands) 0.01 hands")
