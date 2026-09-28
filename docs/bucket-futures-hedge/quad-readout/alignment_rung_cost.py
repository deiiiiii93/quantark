"""What does a fourth refinement rung actually cost the audit?

The audit reports its own price-call count, so this is a measurement rather
than an estimate. Compares the production three-rung ladder against a
four-rung one on a real state, under both readouts.
"""
from datetime import date as date_cls
import json
import os
import sys
from pathlib import Path

REPO = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge")
STUDY = REPO / "example/snowball_q_term_structure"
os.environ["PYTHONPATH"] = str(REPO)
for _k in [k for k in list(sys.modules) if k == "quantark" or k.startswith("quantark.")]:
    del sys.modules[_k]
sys.path[:0] = [str(STUDY), str(REPO)]

import pandas as pd
import quantark

assert str(REPO) in quantark.__file__
import _common as C
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.asset.equity.param import QuadParams
from quantark.backtest.futures_risk import CarryRiskSettings
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.backtest.replay.carry_risk import (
    aggregate_book_risk, audit_held_book, measure_product_carry_risk,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

RUNS = STUDY / "data/bucket_hedge_v2/gate_e/runs"
HIST = Path("/Users/fuxinyao/quant-ark/example/mo_volmodels/data/history")
INCEPTION, MODEL, DAY = "20250401", "term_flat_q", "2025-06-30"
LADDERS = {
    "three rungs (production)": (0.001, 0.0005, 0.00025),
    "four rungs": (0.001, 0.0005, 0.00025, 0.000125),
}

history = C.load_history(HIST)
cell = next(p for p in (RUNS / INCEPTION).iterdir()
            if p.is_dir() and p.name.startswith(MODEL))
summary = json.loads((cell / "run_summary.json").read_text())
config = json.loads((cell / "run_config.json").read_text())
states = pd.read_csv(cell / "states.csv").set_index("date")
legs = pd.read_csv(cell / "hedge_legs.csv")
risk_cfg = config["risk"]
state = states.loc[DAY]

terms = C.build_terms(date_cls.fromisoformat(summary["inception"]),
                      C.TradingCalendar.from_frames(history))
original = C.build_product(terms, summary["s0"], summary["coupon"])
quotes = tuple(IndexFuturesQuote(
    contract=r["contract"], maturity=float(r["tenor_years"]),
    price=float(r["price"]), multiplier=float(r["multiplier"]))
    for r in legs[(legs.date == DAY) & legs.is_curve_node].to_dict("records"))
context = CarryCurveContext(
    quotes, float(state.spot), FlatRateCurve(rate=float(state.rate)),
    config["extrapolation"], "CSI1000", pd.Timestamp(DAY))


def env(spot, dividend):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=float(state.volatility)),
        rate_curve=context.rate_curve, div_yield=dividend,
        valuation_date=pd.Timestamp(DAY).to_pydatetime())


tracker = AutocallableLifecycleTracker(
    product=original, quantity=-1.0, start_date=pd.Timestamp(summary["inception"]))
tracker.lifecycle.knocked_in = bool(state.knocked_in)
product = tracker.product_for_pricing(
    pd.Timestamp(DAY), env(context.spot, context.dividend()))

print(f"{INCEPTION} {MODEL} {DAY}, spot {float(state.spot):.2f}, "
      f"{len(quotes)} curve nodes\n")
print(f"{'readout':14} {'ladder':26} {'audit price calls':>18} {'|R|+E':>10} {'status':>13}")
for readout in ("legacy_linear", "transition"):
    for label, ladder in LADDERS.items():
        settings = CarryRiskSettings(
            reference_notional=float(risk_cfg["reference_notional"]),
            delta_tolerance_hands=float(risk_cfg["delta_tolerance_hands"]),
            rhoq_tolerance_bp=float(risk_cfg["rhoq_tolerance_bp"]),
            futures_bump_points=float(risk_cfg["futures_bump_points"]),
            audit_yield_bump=float(risk_cfg["audit_yield_bump"]),
            identity_spot_bumps_rel=ladder,
            hedge_resolution_rel=risk_cfg.get("hedge_resolution_rel"),
        ).resolved(reference_notional=float(risk_cfg["reference_notional"]),
                   audit_spot_bump_rel=0.01)
        engine = SnowballQuadEngine(QuadParams(
            grid_points=config["quad_grid"], readout=readout,
            align_cell_stretch=0.02))

        def price_at(spot, dividend):
            return -engine.price(product, env(spot, dividend))

        book = aggregate_book_risk([(1.0, measure_product_carry_risk(
            price_at, context, delta_q=0.0,
            points=settings.futures_bump_points))])
        result = audit_held_book(price_at, context, book, {}, settings=settings)
        gate = (abs(result.identity_residual_hands)
                + result.identity_spot_refinement_error_hands)
        print(f"{readout:14} {label:26} {result.price_calls:18d} {gate:10.6f} "
              f"{result.identity_status:>13}")
