"""
Portfolio VaR demonstration.

This example demonstrates the usage of the VaR module with a simple equity portfolio.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    HistoricalVaREngine,
    MonteCarloVaREngine,
    VaRReportGenerator,
)


def create_sample_portfolio():
    """Create a sample equity options portfolio."""
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
        portfolio_name="Sample Options Portfolio",
        pricing_environments={"AAPL": pricing_env},
    )

    call_option = EuropeanVanillaOption(
        strike=100.0, exercise_date=datetime(2024, 7, 1), option_type=OptionType.CALL
    )

    put_option = EuropeanVanillaOption(
        strike=95.0, exercise_date=datetime(2024, 7, 1), option_type=OptionType.PUT
    )

    engine = BlackScholesEngine()

    portfolio.add_position(
        product=call_option,
        quantity=100,
        entry_price=10.5,
        underlying="AAPL",
        engine=engine,
    )

    portfolio.add_position(
        product=put_option,
        quantity=-50,
        entry_price=8.2,
        underlying="AAPL",
        engine=engine,
    )

    return portfolio


def generate_historical_data(num_days=300):
    """Generate synthetic historical market data."""
    np.random.seed(42)
    dates = pd.date_range(start=datetime(2023, 1, 1), periods=num_days, freq="D")

    data = pd.DataFrame(
        {
            "spot_return": np.random.normal(0.0005, 0.015, num_days),
            "vol_change": np.random.normal(0.0, 0.01, num_days),
            "rate_shift": np.random.normal(0.0, 0.0005, num_days),
        },
        index=dates,
    )

    return data


def main():
    """Run VaR calculations and display results."""
    print("=" * 80)
    print("Portfolio VaR Demonstration")
    print("=" * 80)

    portfolio = create_sample_portfolio()
    historical_data = generate_historical_data()

    print(f"\nPortfolio: {portfolio.portfolio_name}")
    print(f"Number of positions: {len(portfolio.positions)}")
    print(f"Portfolio value: ${portfolio.get_portfolio_value():,.2f}")

    equity_factors = EquityRiskFactorConfig(
        include_spot=True, include_vol=True, include_rate=True, include_div_yield=False
    )

    print("\n" + "=" * 80)
    print("1. Parametric VaR (Variance-Covariance)")
    print("=" * 80)

    parametric_config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        lookback_days=252,
        var_method=VaRMethod.PARAMETRIC,
        equity_factors=equity_factors,
        calculate_factor_var=True,
    )

    parametric_engine = ParametricVaREngine(config=parametric_config)
    parametric_result = parametric_engine.calculate_var(portfolio, historical_data)

    summary = VaRReportGenerator.generate_summary(parametric_result)
    for key, value in summary.items():
        print(f"{key:30s}: {value}")

    if parametric_result.factor_var:
        print("\nFactor VaR Attribution:")
        factor_report = VaRReportGenerator.generate_factor_report(parametric_result)
        print(factor_report)

    print("\n" + "=" * 80)
    print("2. Historical VaR (Full Revaluation)")
    print("=" * 80)

    historical_config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        lookback_days=252,
        var_method=VaRMethod.HISTORICAL,
    )

    historical_engine = HistoricalVaREngine(config=historical_config)
    historical_result = historical_engine.calculate_var(portfolio, historical_data)

    summary = VaRReportGenerator.generate_summary(historical_result)
    for key, value in summary.items():
        print(f"{key:30s}: {value}")

    print(f"\nWorst 5 scenarios (P&L):")
    for i, scenario in enumerate(historical_result.worst_scenarios[:5], 1):
        print(f"  {i}. Scenario {scenario['scenario_idx']:3d}: ${scenario['pnl']:,.2f}")

    print("\n" + "=" * 80)
    print("3. Monte Carlo VaR")
    print("=" * 80)

    mc_config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        var_method=VaRMethod.MONTE_CARLO,
        mc_num_simulations=10000,
        mc_seed=42,
    )

    mc_engine = MonteCarloVaREngine(config=mc_config)
    mc_result = mc_engine.calculate_var(portfolio, historical_data)

    summary = VaRReportGenerator.generate_summary(mc_result)
    for key, value in summary.items():
        print(f"{key:30s}: {value}")

    print("\n" + "=" * 80)
    print("VaR Comparison Summary")
    print("=" * 80)

    comparison = pd.DataFrame(
        {
            "Method": ["Parametric", "Historical", "Monte Carlo"],
            "VaR": [parametric_result.var, historical_result.var, mc_result.var],
            "CVaR": [parametric_result.cvar, historical_result.cvar, mc_result.cvar],
            "VaR %": [
                parametric_result.var_as_pct * 100,
                historical_result.var_as_pct * 100,
                mc_result.var_as_pct * 100,
            ],
            "Time (s)": [
                parametric_result.execution_time_seconds,
                historical_result.execution_time_seconds,
                mc_result.execution_time_seconds,
            ],
        }
    )

    print(comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("Demonstration Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
