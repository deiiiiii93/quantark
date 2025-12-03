#!/usr/bin/env python
"""Debug Component VaR calculation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from datetime import datetime

from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from portfolio.equity.portfolio import EquityPortfolio
from priceenv import PricingEnvironment
from util.enum.option_enums import OptionType
from var import VaRConfig, VaRMethod, EquityRiskFactorConfig, ParametricVaREngine

def create_sample_portfolio():
    """Create a sample equity options portfolio for parametric VaR."""
    valuation_date = datetime(2024, 1, 1)

    spot_quote = SpotQuote(spot=100.0, timestamp=valuation_date)
    vol_surface = FlatVolSurface(volatility=0.25)
    rate_curve = FlatRateCurve(rate=0.05)
    div_yield = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=valuation_date,
    )

    portfolio = EquityPortfolio(
        portfolio_name="Parametric VaR Demo Portfolio",
        pricing_environments={"AAPL": pricing_env},
    )

    call_95 = EuropeanVanillaOption(
        strike=95.0, exercise_date=datetime(2024, 6, 1), option_type=OptionType.CALL
    )

    call_105 = EuropeanVanillaOption(
        strike=105.0, exercise_date=datetime(2024, 6, 1), option_type=OptionType.CALL
    )

    put_90 = EuropeanVanillaOption(
        strike=90.0, exercise_date=datetime(2024, 9, 1), option_type=OptionType.PUT
    )

    engine = BlackScholesEngine()

    portfolio.add_position(
        product=call_95,
        quantity=100,
        entry_price=12.5,
        underlying="AAPL",
        engine=engine,
    )

    portfolio.add_position(
        product=call_105,
        quantity=-50,
        entry_price=7.2,
        underlying="AAPL",
        engine=engine,
    )

    portfolio.add_position(
        product=put_90,
        quantity=75,
        entry_price=9.8,
        underlying="AAPL",
        engine=engine,
    )

    return portfolio

def main():
    import pandas as pd

    portfolio = create_sample_portfolio()

    # Generate market data
    np.random.seed(42)
    historical_data = pd.DataFrame({
        "spot_return": np.random.normal(0.0005, 0.015, 300),
        "vol_change": np.random.normal(0.0, 0.01, 300),
        "rate_shift": np.random.normal(0.0, 0.0005, 300),
    })

    # Get sensitivities
    equity_factors = EquityRiskFactorConfig(
        include_spot=True, include_vol=True, include_rate=True, include_div_yield=False
    )

    config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        lookback_days=252,
        var_method=VaRMethod.PARAMETRIC,
        equity_factors=equity_factors,
        calculate_factor_var=True,
        calculate_component_var=True,
        calculate_marginal_var=True,
        calculate_incremental_var=True,
        calculate_stressed_var=True,
    )

    engine = ParametricVaREngine(config=config)

    # Calculate VaR
    result = engine.calculate_var(portfolio, historical_data)

    print("=" * 80)
    print("DEBUG COMPONENT VaR CALCULATION")
    print("=" * 80)
    print()

    # Manually calculate to understand
    print("MANUAL CALCULATION:")
    print("-" * 80)

    # Get position values (market value)
    position_values = {}
    position_sensitivities = {}
    from asset.equity.riskmeasures import GreeksCalculator
    calculator = GreeksCalculator()

    for pos_id, position in portfolio.positions.items():
        pricing_env = portfolio.pricing_environments[position.underlying]
        # Get market value
        option_price = position.engine.price(position.product, pricing_env)
        pos_value = option_price * position.quantity
        position_values[pos_id] = pos_value

        # Get sensitivity
        greeks = calculator.calculate_analytical_greeks(position.product, pricing_env)
        sensitivity = greeks['delta'] * pricing_env.spot
        position_sensitivities[pos_id] = sensitivity

        print(f"Position: {pos_id[:8]}...")
        print(f"  Option Price: ${option_price:8.2f}")
        print(f"  Quantity: {position.quantity:8.0f}")
        print(f"  Market Value: ${pos_value:8.2f}")
        print(f"  Delta: {greeks['delta']:8.4f}")
        print(f"  Sensitivity (δ×spot): {sensitivity:8.2f}")
        print()

    print("PORTFOLIO:")
    print("-" * 80)
    total_value = sum(position_values.values())
    total_sensitivity = sum(position_sensitivities.values())
    print(f"Total Market Value: ${total_value:8.2f}")
    print(f"Total Sensitivity: {total_sensitivity:8.2f}")
    print(f"Portfolio VaR: ${result.var:8.2f}")
    print()

    print("EXPECTED COMPONENT VaR (using proper Euler):")
    print("-" * 80)
    print("Formula: Component VaR_i = sensitivity_i / sum(sensitivities) × Portfolio VaR")
    print()
    expected_sum = 0
    for pos_id in position_values.keys():
        sensitivity = position_sensitivities[pos_id]
        expected_comp_var = abs(sensitivity / total_sensitivity * result.var) if total_sensitivity != 0 else 0
        expected_sum += expected_comp_var
        print(f"  {pos_id[:8]}... | Sensitivity: {sensitivity:8.2f} | Comp VaR: ${expected_comp_var:8.2f}")
    print()
    print(f"Expected Sum: ${expected_sum:8.2f}")
    print(f"Actual Sum: ${sum(result.component_var.values()):8.2f}")
    print()

if __name__ == "__main__":
    main()
