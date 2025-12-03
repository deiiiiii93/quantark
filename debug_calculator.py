#!/usr/bin/env python
"""Debug Component VaR - check inputs to calculator."""

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
from var.attribution import ComponentVaRCalculator
import pandas as pd

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
    portfolio = create_sample_portfolio()

    # Generate market data
    np.random.seed(42)
    historical_data = pd.DataFrame({
        "spot_return": np.random.normal(0.0005, 0.015, 300),
        "vol_change": np.random.normal(0.0, 0.01, 300),
        "rate_shift": np.random.normal(0.0, 0.0005, 300),
    })

    # Get the last 252 days
    risk_factors_df = historical_data.tail(252)
    cov_matrix = risk_factors_df.cov()

    # Get position values and sensitivities (same as in parametric.py)
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

    print("=" * 80)
    print("INPUTS TO ComponentVaRCalculator")
    print("=" * 80)
    print()

    print("Position Values:")
    for pos_id, value in position_values.items():
        print(f"  {pos_id[:8]}... : ${value:8.2f}")
    print()

    print("Position Sensitivities:")
    for pos_id, sens in position_sensitivities.items():
        print(f"  {pos_id[:8]}... : {sens:8.2f}")
    print()

    print("Covariance Matrix:")
    print(cov_matrix)
    print()

    # Calculate Component VaR - use full 3x3 matrix
    # position_sensitivities has position UUIDs as keys
    # factor_sensitivities has factor names as keys
    # We need to calculate portfolio-level factor sensitivities for the matrix index

    # Get portfolio-level sensitivities
    portfolio_sensitivities = {}
    if True:  # include_spot
        total_delta = sum([position_sensitivities[pos_id] for pos_id in position_sensitivities.keys()])
        portfolio_sensitivities['spot_return'] = total_delta
    if False:  # include_vol
        portfolio_sensitivities['vol_change'] = 0.0
    if False:  # include_rate
        portfolio_sensitivities['rate_shift'] = 0.0

    # Calculate Component VaR with full covariance matrix
    comp_calc = ComponentVaRCalculator()
    component_var = comp_calc.calculate_from_sensitivities(
        position_values=position_values,
        sensitivities=position_sensitivities,
        covariance_matrix=pd.DataFrame(
            cov_matrix,
            index=list(portfolio_sensitivities.keys()),
            columns=list(portfolio_sensitivities.keys())
        ),
        confidence_level=0.99
    )

    print("Component VaR Results:")
    for pos_id, cvar in component_var.items():
        print(f"  {pos_id[:8]}... : ${cvar:8.2f}")
    print()

    print(f"Sum of Component VaRs: ${sum(component_var.values()):8.2f}")
    print()

    # Calculate expected manually
    print("Manual Calculation (using Euler decomposition):")
    print("-" * 80)

    from scipy.stats import norm
    z_score = norm.ppf(0.99)

    s_vector = np.array([position_sensitivities[pos_id] for pos_id in position_values.keys()])
    total_sensitivity = np.sum(s_vector)
    primary_factor_var = cov_matrix.iloc[0, 0]
    portfolio_variance = (total_sensitivity ** 2) * primary_factor_var
    portfolio_std = np.sqrt(portfolio_variance)
    portfolio_var = z_score * portfolio_std

    print(f"Total Sensitivity: {total_sensitivity:8.2f}")
    print(f"Factor Variance: {primary_factor_var:8.6f}")
    print(f"Portfolio Variance: {portfolio_variance:8.6f}")
    print(f"Portfolio Std: {portfolio_std:8.6f}")
    print(f"Portfolio VaR: {portfolio_var:8.2f}")
    print()

    expected_sum = 0
    for i, pos_id in enumerate(position_values.keys()):
        sensitivity = position_sensitivities[pos_id]
        covariance = sensitivity * total_sensitivity * primary_factor_var
        allocation_ratio = covariance / portfolio_variance
        component_var_manual = abs(allocation_ratio * portfolio_var)
        expected_sum += component_var_manual
        print(f"  Position {i+1}:")
        print(f"    Sensitivity: {sensitivity:8.2f}")
        print(f"    Covariance: {covariance:8.6f}")
        print(f"    Allocation: {allocation_ratio:8.6f}")
        print(f"    Comp VaR: ${component_var_manual:8.2f}")
        print()

    print(f"Expected Sum: ${expected_sum:8.2f}")

if __name__ == "__main__":
    main()
