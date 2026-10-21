"""Cash-leg folding benchmark: event route vs folded TOTAL vs folded DRILLDOWN trade risk.

One Snowball carrying three cash legs (a margin return leg, a truncating accrual rebate,
a knock-out fee) is risked with the full greek set on each engine, three ways:

- ``event``     — today's route: one ``price_with_events`` solve per bump, legs valued
                  on the event distribution;
- ``TOTAL``     — ``EquityPosition(cash_leg_folding=FoldMode.TOTAL)``: one folded solve
                  per bump, trade totals only;
- ``DRILLDOWN`` — product, total and per-leg greeks from folded solves.

The three arms of one engine run back to back in this process, best of ``--repeat``
wall-clock runs each, so a shared machine's load moves them together.

    python example/cash_leg_folding_benchmark.py            # default grids
    python example/cash_leg_folding_benchmark.py --quick    # tiny grids (smoke)
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

import numpy as np

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2.engine import SnowballQuadEngineV2
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.param.quad_v2_params import QuadV2Params
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.cashleg import (
    AccrualLeg,
    AutocallableCashLeg,
    AutocallableLegType,
    BaseAmount,
    BaseAmountMode,
    FixedPayoffLeg,
    FoldMode,
    KOBehavior,
    LegDirection,
    LegSchedule,
    PaymentTrigger,
    PvFormula,
)
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.equity.position import EquityPosition
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

GREEKS = ["delta", "gamma", "vega", "rho", "dividend_rho", "theta"]
KO_DATES = tuple(round(0.25 * k, 2) for k in range(1, 5))
NOTIONAL = 1_000_000.0


def make_env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2024, 1, 1),
    )


def make_snowball() -> SnowballOption:
    barrier = BarrierConfig(
        ko_barrier=103.0, ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(KO_DATES),
        ki_barrier=75.0, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True,
    )
    return SnowballOption(initial_price=100.0, strike=100.0, barrier_config=barrier,
                          contract_multiplier=1.0, maturity=1.0, is_reverse=False)


def make_legs():
    starts = np.array([0.0, 0.25, 0.5, 0.75])
    return [
        AutocallableCashLeg(
            direction=LegDirection.BUYER_PAYS, leg_type=AutocallableLegType.MARGIN,
            notional=NOTIONAL, rate=0.05, observation_schedule=KO_DATES,
            accrual_factors=KO_DATES, settlement_schedule=KO_DATES,
            terminal_accrual_factor=1.0, terminal_settlement_time=1.0,
            pv_formula=PvFormula.NOTIONAL_MINUS_PAYOFF, name="margin",
        ),
        AccrualLeg(
            direction=LegDirection.BUYER_RECEIVES, rate=0.02,
            base=BaseAmount(value=NOTIONAL, mode=BaseAmountMode.ABSOLUTE),
            schedule=LegSchedule(period_starts=starts, period_ends=starts + 0.25,
                                 payment_times=starts + 0.25),
            ko_behavior=KOBehavior.TRUNCATE_AT_KO, name="rebate",
        ),
        FixedPayoffLeg(direction=LegDirection.BUYER_PAYS, amount=2_500.0,
                       trigger=PaymentTrigger.AT_KO, name="ko fee"),
    ]


def engines(quick: bool):
    pde_points = dict(accuracy="fast") if quick else {}
    return {
        "QUAD v1": lambda: SnowballQuadEngine(params=QuadParams(grid_points=301 if quick else 1001)),
        "QUAD v2": lambda: SnowballQuadEngineV2(QuadV2Params()),
        "PDE 1D": lambda: SnowballPDESolver(params=PDEParams(**pde_points)),
        "MC": lambda: SnowballMCEngine(params=MCParams(num_paths=8_192 if quick else 100_000, seed=42)),
    }


def risk(engine_factory, mode):
    pos = EquityPosition(product=make_snowball(), quantity=1.0, entry_price=0.0, underlying="X",
                         engine=engine_factory(), entry_timestamp=datetime(2024, 1, 1),
                         cash_legs=make_legs(), cash_leg_folding=mode)
    return pos.get_trade_risk(make_env(), GreeksCalculator(), GREEKS)


def best_time(fn, repeat):
    best, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def run(quick: bool = False, repeat: int = 3):
    """One row per engine: wall-clock seconds of each arm and the largest total-greek gap."""
    rows = []
    for name, factory in engines(quick).items():
        t_event, event = best_time(lambda: risk(factory, None), repeat)
        t_total, total = best_time(lambda: risk(factory, FoldMode.TOTAL), repeat)
        t_drill, drill = best_time(lambda: risk(factory, FoldMode.DRILLDOWN), repeat)
        gap = max(abs(total.total[k] - event.total[k]) for k in event.total)
        rows.append(dict(engine=name, event=t_event, total=t_total, drilldown=t_drill,
                         speedup_total=t_event / t_total, speedup_drilldown=t_event / t_drill,
                         max_total_gap=gap, drill_total_gap=max(abs(drill.total[k] - total.total[k])
                                                               for k in total.total)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="tiny grids / path counts")
    parser.add_argument("--repeat", type=int, default=3, help="best-of-N wall clock per arm")
    args = parser.parse_args()
    rows = run(quick=args.quick, repeat=args.repeat)
    header = f"{'engine':<9}{'event s':>10}{'TOTAL s':>10}{'DRILL s':>10}{'×TOTAL':>9}{'×DRILL':>9}{'max |Δtotal|':>15}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['engine']:<9}{r['event']:>10.3f}{r['total']:>10.3f}{r['drilldown']:>10.3f}"
              f"{r['speedup_total']:>9.2f}{r['speedup_drilldown']:>9.2f}{r['max_total_gap']:>15.3e}")


if __name__ == "__main__":
    main()
