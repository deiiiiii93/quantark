"""
Demonstration of BarrierAnalyticalEngine for single-barrier options.

This script shows:
- Continuous knock-out pricing (closed-form)
- Discrete knock-out pricing using barrier shift (regular observation grid)
- Expiry-only monitoring decomposed into vanillas + digitals (rebate supported)
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import QuantArk modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.engine.analytical import BarrierAnalyticalEngine
from asset.equity.engine.mc import BarrierOptionMCEngine
from asset.equity.param import MCParams
from asset.equity.product.option import (
    BarrierOption,
    ObservationRecord,
    ObservationSchedule,
)
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import BarrierType, ObservationType, OptionType


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def make_pricing_env() -> PricingEnvironment:
    """Create a simple flat market environment."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0, asset_name="XYZ"),
        vol_surface=FlatVolSurface(volatility=0.25),  # 25% vol
        rate_curve=FlatRateCurve(rate=0.03),  # 3% risk-free
        div_yield=ContinuousDividendYield(div_yield=0.01),  # 1% dividend
        valuation_date=datetime(2024, 1, 1),
    )


def demo_continuous_knock_out(engine: BarrierAnalyticalEngine, mc_engine: BarrierOptionMCEngine, env: PricingEnvironment):
    """Continuous DOWN&OUT call, no rebate."""
    print_section("Continuous monitoring (Down-and-Out Call)")
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=90.0,
        barrier_type=BarrierType.DOWN_OUT,
        maturity=1.0,
        observation_type=ObservationType.CONTINUOUS,
    )
    price = engine.price(option, env)
    mc_price = mc_engine.price(option, env)
    print(f"Price (continuous down-out call) [Analytic]: {price:.6f}")
    print(f"Price (continuous down-out call) [MC]      : {mc_price:.6f}")


def demo_daily_knock_out(engine: BarrierAnalyticalEngine, mc_engine: BarrierOptionMCEngine, env: PricingEnvironment):
    """Discrete DOWN&OUT call with daily observations, no rebate."""
    print_section("Discrete monitoring with daily observations (Down-and-Out Call)")
    daily = 1.0 / 245.0
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(observation_time=i * daily, barrier=90.0, payoff=0.0)
            for i in range(1, 245)
        ],
        frequency=daily,
    )
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=90.0,
        barrier_type=BarrierType.DOWN_OUT,
        maturity=1.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
    )
    price = engine.price(option, env)
    mc_price = mc_engine.price(option, env)
    print(f"Price (daily down-out call) [Analytic]: {price:.6f}")
    print(f"Price (daily down-out call) [MC]      : {mc_price:.6f}")


def demo_discrete_knock_out(engine: BarrierAnalyticalEngine, mc_engine: BarrierOptionMCEngine, env: PricingEnvironment):
    """Discrete UP&OUT call with monthly observations (barrier shift applied)."""
    print_section("Discrete monitoring with barrier shift (Up-and-Out Call)")
    monthly = 1.0 / 12.0
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(observation_time=i * monthly, barrier=110.0, payoff=0.0)
            for i in range(1, 13)
        ],
        frequency=monthly,
    )
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
    )
    price = engine.price(option, env)
    mc_price = mc_engine.price(option, env)
    print(f"Price (discrete up-out call, monthly obs) [Analytic]: {price:.6f}")
    print(f"Price (discrete up-out call, monthly obs) [MC]      : {mc_price:.6f}")


def demo_expiry_with_rebate(engine: BarrierAnalyticalEngine, mc_engine: BarrierOptionMCEngine, env: PricingEnvironment):
    """Expiry-only UP&OUT call with rebate, uses vanilla + digital decomposition."""
    print_section("Expiry-only monitoring (Up-and-Out Call with rebate)")
    option = BarrierOption(
        strike=95.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        observation_type=ObservationType.EXPIRY,
        rebate=2.0,
    )
    price = engine.price(option, env)
    mc_price = mc_engine.price(option, env)
    print(f"Price (expiry up-out call with rebate) [Analytic]: {price:.6f}")
    print(f"Price (expiry up-out call with rebate) [MC]      : {mc_price:.6f}")


def main():
    env = make_pricing_env()
    engine = BarrierAnalyticalEngine()
    
    # Setup MC engine with enough paths and steps for reasonable accuracy
    mc_params = MCParams(num_paths=50000, time_steps=252, seed=42)
    mc_engine = BarrierOptionMCEngine(params=mc_params)

    demo_continuous_knock_out(engine, mc_engine, env)
    demo_daily_knock_out(engine, mc_engine, env)
    demo_discrete_knock_out(engine, mc_engine, env)
    demo_expiry_with_rebate(engine, mc_engine, env)


if __name__ == "__main__":
    main()
