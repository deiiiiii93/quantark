"""End-to-end demo: snowball with premium, accrual interest, and KO bonus.

Run: python example/cash_legs_demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.engine.quad import SnowballQuadEngine
from asset.equity.param import QuadParams
from asset.equity.product.option.snowball_helpers import create_standard_snowball
from cashleg import (
    AccrualLeg,
    BaseAmount,
    BaseAmountMode,
    DeterministicLeg,
    FixedPayoffLeg,
    KOBehavior,
    LegDirection,
    LegSchedule,
    PaymentConvention,
    PaymentTrigger,
    SurvivalBasis,
)
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from portfolio.equity.position import EquityPosition
from priceenv import PricingEnvironment
from util.calendar.day_counter import DayCountConvention


def main() -> None:
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )
    snowball = create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        num_observations=4,
        ko_barrier=103.0,
        ko_rate=0.15,
        ki_barrier=75.0,
        contract_multiplier=10_000.0,
    )
    engine = SnowballQuadEngine(params=QuadParams(grid_points=201))

    quarterly_schedule = LegSchedule(
        period_starts=np.array([0.0, 0.25, 0.5, 0.75]),
        period_ends=np.array([0.25, 0.5, 0.75, 1.0]),
        payment_times=np.array([0.25, 0.5, 0.75, 1.0]),
    )

    position = EquityPosition(
        product=snowball,
        quantity=1.0,
        entry_price=engine.price(snowball, env),
        underlying="CSI300",
        engine=engine,
        entry_timestamp=datetime(2026, 1, 1),
        cash_legs=[
            DeterministicLeg(
                amount=150_000.0,
                payment_time=0.0,
                direction=LegDirection.BUYER_PAYS,
                name="Front Premium",
            ),
            AccrualLeg(
                rate=0.02,
                base=BaseAmount(value=1.0, mode=BaseAmountMode.NOTIONAL_FRACTION),
                schedule=quarterly_schedule,
                day_count=DayCountConvention.ACT_365,
                payment_convention=PaymentConvention.AT_PERIOD_END,
                ko_behavior=KOBehavior.TRUNCATE_AT_KO,
                survival_basis=SurvivalBasis.ENTER_PERIOD,
                direction=LegDirection.BUYER_RECEIVES,
                name="Margin Interest",
            ),
            FixedPayoffLeg(
                amount=50_000.0,
                trigger=PaymentTrigger.AT_KO,
                direction=LegDirection.BUYER_RECEIVES,
                name="KO Bonus",
            ),
        ],
    )

    breakdown = position.get_trade_value_breakdown(env)
    print(f"Product NPV (per unit): {engine.price(snowball, env):,.2f}")
    print(f"Position market value (product only): {position.get_market_value(env):,.2f}")
    print(f"Position trade value (product + legs): {breakdown.total:,.2f}")
    print()
    print("Trade value breakdown:")
    print(f"  Product NPV       : {breakdown.product_npv:>15,.2f}")
    for leg_pv in breakdown.leg_pvs.values():
        print(
            f"  {leg_pv.name:<18}: {leg_pv.pv:>15,.2f}  ({leg_pv.direction.name})"
        )
    print(f"  {'TOTAL':<18}: {breakdown.total:>15,.2f}")


if __name__ == "__main__":
    main()
