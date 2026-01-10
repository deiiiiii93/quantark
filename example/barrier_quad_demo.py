"""
Demonstration of single-barrier option pricing using quadrature.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.engine.analytical import BarrierAnalyticalEngine
from asset.equity.engine.quad import BarrierQuadEngine
from asset.equity.engine.mc import BarrierOptionMCEngine
from asset.equity.param import QuadParams, MCParams
from asset.equity.product.option import BarrierOption
from asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.enum import BarrierType, ObservationAggregation, ObservationType, OptionType


def create_pricing_env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def build_schedule(barrier: float, rebate: float) -> ObservationSchedule:
    return ObservationSchedule(
        records=[
            ObservationRecord(observation_time=0.25, barrier=barrier, payoff=rebate),
            ObservationRecord(observation_time=0.50, barrier=barrier, payoff=rebate),
            ObservationRecord(observation_time=0.75, barrier=barrier, payoff=rebate),
            ObservationRecord(observation_time=1.00, barrier=barrier, payoff=rebate),
        ],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
    )


def build_daily_schedule(barrier: float, rebate: float) -> ObservationSchedule:
    """Build a daily observation schedule (252 days)."""
    records = []
    for i in range(1, 253):
        t = i / 252.0
        records.append(
            ObservationRecord(observation_time=t, barrier=barrier, payoff=rebate)
        )
    return ObservationSchedule(
        records=records,
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
    )


def main() -> int:
    env = create_pricing_env()

    # Quarterly Schedules
    schedule_up = build_schedule(barrier=110.0, rebate=2.0)
    schedule_down = build_schedule(barrier=90.0, rebate=2.0)

    # Daily Schedules
    schedule_up_daily = build_daily_schedule(barrier=110.0, rebate=2.0)
    schedule_down_daily = build_daily_schedule(barrier=90.0, rebate=2.0)

    # Quarterly Options
    up_out = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=2.0,
        pay_at_hit=True,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule_up,
    )
    down_in = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=90.0,
        barrier_type=BarrierType.DOWN_IN,
        maturity=1.0,
        rebate=2.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule_down,
    )

    # Daily Options
    up_out_daily = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=2.0,
        pay_at_hit=True,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule_up_daily,
    )
    down_in_daily = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=90.0,
        barrier_type=BarrierType.DOWN_IN,
        maturity=1.0,
        rebate=2.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule_down_daily,
    )

    quad_engine = BarrierQuadEngine(params=QuadParams(grid_points=1601))
    analytical_engine = BarrierAnalyticalEngine()
    
    mc_params = MCParams(num_paths=200000, time_steps=252, seed=42)
    mc_engine = BarrierOptionMCEngine(params=mc_params)

    print("Barrier Quad Demo")
    print("-" * 90)
    print(f"{'Option':<25} {'Quad Price':>12} {'Analytic Price':>15} {'MC Price':>12} {'Diff (Q-A)':>12}")
    print("-" * 90)

    test_cases = [
        ("Up-and-Out (Qtly)", up_out),
        ("Up-and-Out (Daily)", up_out_daily),
        ("Down-and-In (Qtly)", down_in),
        ("Down-and-In (Daily)", down_in_daily),
    ]

    for label, option in test_cases:
        quad_price = quad_engine.price(option, env)
        analytical_price = analytical_engine.price(option, env)
        mc_price = mc_engine.price(option, env)
        diff = quad_price - analytical_price
        print(f"{label:<25} {quad_price:>12.6f} {analytical_price:>15.6f} {mc_price:>12.6f} {diff:>12.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
