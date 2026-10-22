"""
Regression demo comparing OneTouchQuadEngine vs OneTouchAnalyticalEngine
for expiry-only monitoring.
"""

import sys
from datetime import datetime
from pathlib import Path


from quantark.asset.equity.engine.analytical import OneTouchAnalyticalEngine
from quantark.asset.equity.engine.quad import OneTouchQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option import OneTouchOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import BarrierDirection, ObservationType, TouchType


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.2,
    rate: float = 0.03,
    div: float = 0.01,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def print_case(option: OneTouchOption, quad_price: float, ana_price: float) -> None:
    direction = "Up" if option.is_up_barrier else "Down"
    touch_type = "OneTouch" if option.is_one_touch else "NoTouch"
    if option.observation_type == ObservationType.EXPIRY:
        obs_label = "expiry-only"
    else:
        obs_label = "discrete"
    pay_label = "pay-at-hit" if option.payment_at_hit else "pay-at-expiry"
    print(f"\nCase: {direction} {touch_type} ({obs_label}, {pay_label})")
    print(f"Barrier: {option.barrier:.2f}  Rebate: {option.rebate:.2f}")
    print(f"Analytical: {ana_price:.6f}  Quad: {quad_price:.6f}")
    print(f"Abs Diff: {abs(quad_price - ana_price):.6f}")


def main() -> None:
    env = create_pricing_env()
    quad_engine = OneTouchQuadEngine(params=QuadParams(grid_points=2001))
    ana_engine = OneTouchAnalyticalEngine()

    daily_obs = [i / 252 for i in range(1, 253)]
    cases = [
        OneTouchOption(
            barrier=110.0,
            barrier_direction=BarrierDirection.UP,
            maturity=1.0,
            rebate=5.0,
            payment_at_hit=False,
            touch_type=TouchType.ONE_TOUCH,
            observation_type=ObservationType.EXPIRY,
        ),
        OneTouchOption(
            barrier=90.0,
            barrier_direction=BarrierDirection.DOWN,
            maturity=1.0,
            rebate=5.0,
            payment_at_hit=False,
            touch_type=TouchType.NO_TOUCH,
            observation_type=ObservationType.EXPIRY,
        ),
        OneTouchOption(
            barrier=110.0,
            barrier_direction=BarrierDirection.UP,
            maturity=1.0,
            rebate=5.0,
            payment_at_hit=True,
            touch_type=TouchType.ONE_TOUCH,
            observation_type=ObservationType.DISCRETE,
            observation_dates=[0.25, 0.5, 0.75, 1.0],
        ),
        OneTouchOption(
            barrier=90.0,
            barrier_direction=BarrierDirection.DOWN,
            maturity=1.0,
            rebate=5.0,
            payment_at_hit=False,
            touch_type=TouchType.NO_TOUCH,
            observation_type=ObservationType.DISCRETE,
            observation_dates=[i / 12 for i in range(1, 13)],
        ),
        OneTouchOption(
            barrier=108.0,
            barrier_direction=BarrierDirection.UP,
            maturity=1.0,
            rebate=5.0,
            payment_at_hit=True,
            touch_type=TouchType.ONE_TOUCH,
            observation_type=ObservationType.DISCRETE,
            observation_dates=daily_obs,
        ),
        OneTouchOption(
            barrier=92.0,
            barrier_direction=BarrierDirection.DOWN,
            maturity=1.0,
            rebate=5.0,
            payment_at_hit=False,
            touch_type=TouchType.NO_TOUCH,
            observation_type=ObservationType.DISCRETE,
            observation_dates=daily_obs,
        ),
    ]

    print("OneTouch Quad vs Analytical (Expiry and Discrete Monitoring)")
    for option in cases:
        quad_price = quad_engine.price(option, env)
        ana_price = ana_engine.price(option, env)
        print_case(option, quad_price, ana_price)


if __name__ == "__main__":
    main()
