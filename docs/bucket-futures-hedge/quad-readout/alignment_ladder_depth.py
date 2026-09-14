"""Why can the ladder not resolve 2025-06-30 once the cell is snapped?

The residual there is small (0.0013 hands) but the REFINEMENT is large
(0.0185), meaning the residual is still moving between the last two rungs.
That is a convergence failure of the measurement, not a violated identity.

Prime suspect is the readout staircase, which is a separate defect on the
record: the price is np.interp between nodes, so a spot sitting very close
to a node boundary makes sub-cell bumps straddle a tread edge and the rungs
disagree. Prints the whole ladder and where spot sits in its cell, with the
option off and on.
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

import numpy as np
import pandas as pd
import quantark

assert str(REPO) in quantark.__file__
import _common as C
from quantark.asset.equity.engine.quad import quad_math as QM
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.asset.equity.param import QuadParams
from quantark.backtest.futures_risk import CarryRiskSettings
from quantark.backtest.replay.carry_context import CarryCurveContext
from quantark.backtest.replay.carry_risk import (
    aggregate_book_risk, measure_product_carry_risk, sample_chain_identity,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

RUNS = STUDY / "data/bucket_hedge_v2/gate_e/runs"
CASES = [
    ("20250401", "term_flat_q", "2025-06-30"),
    ("20250401", "term_flat_fwd", "2025-06-30"),
]
LADDER = [float(x) for x in os.environ.get("LADDER", "").split(",") if x]
HIST = Path("/Users/fuxinyao/quant-ark/example/mo_volmodels/data/history")
history = C.load_history(HIST)


def probe(inception, model, day, stretch):
    cell = next(p for p in (RUNS / inception).iterdir()
                if p.is_dir() and p.name.startswith(model))
    summary = json.loads((cell / "run_summary.json").read_text())
    config = json.loads((cell / "run_config.json").read_text())
    states = pd.read_csv(cell / "states.csv").set_index("date")
    legs = pd.read_csv(cell / "hedge_legs.csv")
    risk_cfg = config["risk"]
    settings = CarryRiskSettings(
        reference_notional=float(risk_cfg["reference_notional"]),
        delta_tolerance_hands=float(risk_cfg["delta_tolerance_hands"]),
        rhoq_tolerance_bp=float(risk_cfg["rhoq_tolerance_bp"]),
        futures_bump_points=float(risk_cfg["futures_bump_points"]),
        audit_yield_bump=float(risk_cfg["audit_yield_bump"]),
        identity_spot_bumps_rel=tuple(LADDER or risk_cfg["identity_spot_bumps_rel"]),
    ).resolved(reference_notional=float(risk_cfg["reference_notional"]),
               audit_spot_bump_rel=0.01)

    terms = C.build_terms(date_cls.fromisoformat(summary["inception"]),
                          C.TradingCalendar.from_frames(history))
    original = C.build_product(terms, summary["s0"], summary["coupon"])
    state = states.loc[day]
    node_legs = legs[(legs.date == day) & legs.is_curve_node]
    quotes = tuple(IndexFuturesQuote(
        contract=r["contract"], maturity=float(r["tenor_years"]),
        price=float(r["price"]), multiplier=float(r["multiplier"]))
        for r in node_legs.to_dict("records"))
    context = CarryCurveContext(
        quotes, float(state.spot), FlatRateCurve(rate=float(state.rate)),
        config["extrapolation"], "CSI1000", pd.Timestamp(day))

    def env(spot, dividend):
        return PricingEnvironment(
            spot_quote=SpotQuote(spot=spot),
            vol_surface=FlatVolSurface(volatility=float(state.volatility)),
            rate_curve=context.rate_curve, div_yield=dividend,
            valuation_date=pd.Timestamp(day).to_pydatetime())

    tracker = AutocallableLifecycleTracker(
        product=original, quantity=-1.0,
        start_date=pd.Timestamp(summary["inception"]))
    tracker.lifecycle.knocked_in = bool(state.knocked_in)
    product = tracker.product_for_pricing(
        pd.Timestamp(day), env(context.spot, context.dividend()))

    quad = {"grid_points": config["quad_grid"],
            "readout": config.get("quad_readout", "legacy_linear")}
    if stretch is not None:
        quad["align_cell_stretch"] = float(stretch)
    engine = SnowballQuadEngine(QuadParams(**quad))

    seen = []
    original_init = QM.QuadratureMath.__init__

    def spy(self, *a, **k):
        original_init(self, *a, **k)
        seen.append(self)

    QM.QuadratureMath.__init__ = spy
    try:
        def price_at(spot, dividend):
            return -engine.price(product, env(spot, dividend))

        product_risk = measure_product_carry_risk(
            price_at, context, delta_q=0.0, points=settings.futures_bump_points)
        book = aggregate_book_risk([(1.0, product_risk)])
        samples = sample_chain_identity(price_at, context, book, settings)
    finally:
        QM.QuadratureMath.__init__ = original_init

    math_utils = seen[0]
    # where does spot (log-moneyness 0) sit inside its cell?
    grid = math_utils.grid
    j = int(np.searchsorted(grid, 0.0)) - 1
    frac = (0.0 - grid[j]) / math_utils.h if 0 <= j < len(grid) - 1 else float("nan")
    return {
        "samples": samples,
        "h_pts": math_utils.h * float(state.spot),
        "snap": math_utils.cell_snap_ratio,
        "off_node": math_utils.barrier_cell_offset * float(state.spot),
        "frac": frac,
        "knocked_in": bool(state.knocked_in),
        "spot": float(state.spot),
    }


for inception, model, day in CASES:
    print(f"\n=== {inception} {model} {day} ===")
    for label, stretch in (("off", None), ("on", 0.02)):
        out = probe(inception, model, day, stretch)
        rungs = "  ".join(f"{s.spot_bump_rel:.5f}:{s.residual_hands:+.6f}"
                          for s in out["samples"])
        R = out["samples"][-1].residual_hands
        E = abs(R - out["samples"][-2].residual_hands)
        print(f"  {label:3} spot {out['spot']:.2f} KI={out['knocked_in']} "
              f"h={out['h_pts']:.3f}pts snap={out['snap']:.6f} "
              f"barrier_off_node={out['off_node']:.3e}pts")
        print(f"      spot sits {out['frac']:.4f} of the way through its cell")
        print(f"      ladder  {rungs}")
        print(f"      R={R:+.6f}  E={E:.6f}  |R|+E={abs(R) + E:.6f}")
