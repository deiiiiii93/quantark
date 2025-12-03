#!/usr/bin/env python
"""Debug why Component VaR calculation doesn't match Portfolio VaR."""

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
from scipy import stats

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

    # Calculate engine VaR
    equity_factors = EquityRiskFactorConfig(
        include_spot=True, include_vol=False, include_rate=False, include_div_yield=False
    )

    config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        lookback_days=252,
        var_method=VaRMethod.PARAMETRIC,
        equity_factors=equity_factors,
        calculate_factor_var=True,
        calculate_component_var=True,
        calculate_marginal_var=False,
        calculate_incremental_var=False,
        calculate_stressed_var=False,
    )

    engine = ParametricVaREngine(config=config)
    result = engine.calculate_var(portfolio, historical_data)

    print("=" * 80)
    print("DEBUG VaR CALCULATION MISMATCH")
    print("=" * 80)
    print()
    print(f"Engine Portfolio VaR: ${result.var:8.2f}")
    print()

    # Manually calculate VaR using the same method as the engine
    print("MANUAL CALCULATION (using engine method):")
    print("-" * 80)

    # Get last 252 days
    risk_factors_df = historical_data.tail(252)

    # Calculate sensitivities (same as engine)
    from asset.equity.riskmeasures import GreeksCalculator
    calculator = GreeksCalculator()

    factor_sensitivities = {}
    total_delta = 0.0
    for position in portfolio.positions.values():
        pricing_env = portfolio.pricing_environments[position.underlying]
        greeks = calculator.calculate_analytical_greeks(
            position.product, pricing_env
        )
        total_delta += greeks['delta'] * position.quantity * pricing_env.spot
    factor_sensitivities['spot_return'] = total_delta

    print(f"Total Delta Sensitivity: {total_delta:8.2f}")
    print()

    # Calculate portfolio variance
    cov_matrix = risk_factors_df.cov().values
    sensitivity_vector = np.array([factor_sensitivities['spot_return']])
    portfolio_variance = sensitivity_vector @ cov_matrix @ sensitivity_vector
    portfolio_std = np.sqrt(portfolio_variance)

    z_score = stats.norm.ppf(0.99)
    var = z_score * portfolio_std

    print(f"Covariance Matrix (spot_return only):")
    print(f"  Variance: {cov_matrix[0, 0]:.6f}")
    print()
    print(f"Portfolio Variance: {portfolio_variance:.6f}")
    print(f"Portfolio Std: {portfolio_std:.6f}")
    print(f"Z-Score: {z_score:.4f}")
    print(f"Calculated VaR: ${var:.2f}")
    print()

    # Now calculate Component VaR
    print("COMPONENT VaR CALCULATION:")
    print("-" * 80)

    factor_returns = risk_factors_df[['spot_return']].values
    portfolio_pnl = factor_returns.flatten() * total_delta

    print(f"Portfolio P&L (first 5 scenarios):")
    print(f"  {portfolio_pnl[:5]}")
    print(f"Portfolio P&L std: ${np.std(portfolio_pnl):.2f}")
    print()

    position_pnls = {}
    for pos_id, position in portfolio.positions.items():
        pricing_env = portfolio.pricing_environments[position.underlying]
        greeks = calculator.calculate_analytical_greeks(
            position.product, pricing_env
        )
        pos_sensitivity = greeks['delta'] * pricing_env.spot
        pos_pnl = factor_returns.flatten() * pos_sensitivity
        position_pnls[pos_id] = pos_pnl

        print(f"Position {pos_id[:8]}...")
        print(f"  Sensitivity: {pos_sensitivity:8.2f}")
        print(f"  P&L (first 5): {pos_pnl[:5]}")
        print(f"  P&L std: ${np.std(pos_pnl):.2f}")

    print()

    # Calculate Component VaR
    portfolio_var = np.var(portfolio_pnl, ddof=1)
    portfolio_var_result = z_score * np.sqrt(portfolio_var)

    print(f"Portfolio Variance (from scenarios): {portfolio_var:.6f}")
    print(f"Portfolio VaR (from scenarios): ${portfolio_var_result:.2f}")
    print()

    component_var = {}
    for pos_id, pos_pnl in position_pnls.items():
        covariance = np.cov(pos_pnl, portfolio_pnl, ddof=1)[0, 1]
        contribution_ratio = covariance / portfolio_var
        component_var_result = abs(contribution_ratio * portfolio_var_result)
        component_var[pos_id] = component_var_result
        print(f"Position {pos_id[:8]}...")
        print(f"  Covariance: {covariance:.6f}")
        print(f"  Contribution Ratio: {contribution_ratio:.6f}")
        print(f"  Component VaR: ${component_var_result:.2f}")

    print()
    print(f"Sum of Component VaRs: ${sum(component_var.values()):.2f}")

if __name__ == "__main__":
    main()
