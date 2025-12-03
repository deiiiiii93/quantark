#!/usr/bin/env python
"""Debug script to understand parametric VaR attribution issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

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

from var import (
    VaRConfig,
    VaRMethod,
    EquityRiskFactorConfig,
    ParametricVaREngine,
)

def create_sample_portfolio():
    """Create a sample equity options portfolio for parametric VaR."""
    valuation_date = sys.modules['datetime'].datetime(2024, 1, 1)

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

    # Create a portfolio with multiple positions showing different exposures
    call_95 = EuropeanVanillaOption(
        strike=95.0, exercise_date=sys.modules['datetime'].datetime(2024, 6, 1), option_type=OptionType.CALL
    )

    call_105 = EuropeanVanillaOption(
        strike=105.0, exercise_date=sys.modules['datetime'].datetime(2024, 6, 1), option_type=OptionType.CALL
    )

    put_90 = EuropeanVanillaOption(
        strike=90.0, exercise_date=sys.modules['datetime'].datetime(2024, 9, 1), option_type=OptionType.PUT
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
    print("=" * 80)
    print("DEBUGGING PARAMETRIC VaR ATTRIBUTION")
    print("=" * 80)
    print()

    import numpy as np
    import pandas as pd
    from datetime import datetime

    portfolio = create_sample_portfolio()
    historical_data = pd.DataFrame({
        "spot_return": np.random.normal(0.0005, 0.015, 300),
        "vol_change": np.random.normal(0.0, 0.01, 300),
        "rate_shift": np.random.normal(0.0, 0.0005, 300),
    })

    print("PORTFOLIO POSITIONS:")
    print("-" * 80)
    for pos_id, position in portfolio.positions.items():
        product_name = position.product.__class__.__name__
        strike = position.product.strike
        expiry = position.product.exercise_date.strftime("%Y-%m-%d")
        print(f"  {pos_id[:8]}... | {product_name} | Strike: ${strike:6.1f} | Qty: {position.quantity:4d}")
    print()

    print("POSITION VALUES:")
    print("-" * 80)
    for pos_id, position in portfolio.positions.items():
        pricing_env = portfolio.pricing_environments[position.underlying]
        pos_value = pricing_env.spot * position.quantity
        print(f"  {pos_id[:8]}... | Position Value: ${pos_value:8.2f}")
    print()

    print("SENSITIVITY CALCULATION:")
    print("-" * 80)
    from asset.equity.riskmeasures import GreeksCalculator
    calculator = GreeksCalculator()

    for pos_id, position in portfolio.positions.items():
        pricing_env = portfolio.pricing_environments[position.underlying]
        greeks = calculator.calculate_analytical_greeks(position.product, pricing_env)
        delta = greeks['delta']
        sensitivity = delta * pricing_env.spot
        print(f"  {pos_id[:8]}... | Delta: {delta:8.4f} | Sensitivity (δ×spot): {sensitivity:8.2f}")
    print()

    print("COVARIANCE MATRIX:")
    print("-" * 80)
    print(historical_data.cov())
    print()

    print("EXPECTED COMPONENT VaR (using proper Euler decomposition):")
    print("-" * 80)
    print("  Component VaR_i = Cov(P&L_i, P&L_portfolio) / Var(P&L_portfolio) * Portfolio VaR")
    print("  Or using sensitivities:")
    print("  Component VaR_i = (sensitivity_i × σ_factor) / σ_portfolio × Portfolio VaR")
    print()

    print("ACTUAL BUGGY IMPLEMENTATION:")
    print("-" * 80)
    print("  The current code calculates:")
    print("  var_contribution = z_score × sqrt(sum((sensitivity_i × value_i)^2) × variance)")
    print("  This is the STANDALONE VaR, NOT the component VaR!")
    print()

if __name__ == "__main__":
    main()
