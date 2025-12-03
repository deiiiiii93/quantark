"""
Parametric VaR (Variance-Covariance) demonstration.

This example demonstrates the use of Parametric VaR, which uses portfolio sensitivities
(delta, gamma, vega, rho) to calculate VaR through variance-covariance matrix operations.

Methodology:
- Uses first and second-order sensitivities (Greeks) from pricing models
- Assumes normally distributed risk factor returns
- VaR = z_score * sqrt(s^T * Σ * s) where s = sensitivities, Σ = covariance matrix

Advantages:
- Fastest method (O(f³) for covariance inversion)
- Excellent for large portfolios (100,000+ positions)
- Provides detailed attribution (Component, Factor, Marginal VaR)
- Regulatory standard (Basel III/IV)

Best For:
- Large equity portfolios
- Linear products (stocks, forwards, swaps)
- Real-time risk monitoring
- When Greeks are reliable (near expiry, ATM options)

Limitations:
- Assumes normality (may under/over-estimate tail risk)
- Less accurate for highly non-linear products
- Requires stable covariance matrix
"""

import numpy as np
import pandas as pd
from datetime import datetime

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
    VaRReportGenerator,
)


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

    # Create a portfolio with multiple positions showing different exposures
    # Long call spread (bullish, positive delta, positive gamma)
    call_95 = EuropeanVanillaOption(
        strike=95.0, exercise_date=datetime(2024, 6, 1), option_type=OptionType.CALL
    )

    call_105 = EuropeanVanillaOption(
        strike=105.0, exercise_date=datetime(2024, 6, 1), option_type=OptionType.CALL
    )

    # Short put (bearish, negative delta)
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


def generate_historical_data(num_days=300):
    """Generate synthetic historical market data for parametric VaR."""
    np.random.seed(42)
    dates = pd.date_range(start=datetime(2023, 1, 1), periods=num_days, freq="D")

    # Realistic market data with some correlation
    data = pd.DataFrame(
        {
            "spot_return": np.random.normal(0.0005, 0.015, num_days),
            "vol_change": np.random.normal(0.0, 0.015, num_days),
            "rate_shift": np.random.normal(0.0, 0.0005, num_days),
        },
        index=dates,
    )

    return data


def main():
    """Run Parametric VaR demonstration and display results."""
    print("=" * 80)
    print("Parametric VaR (Variance-Covariance) Demonstration")
    print("=" * 80)
    print()
    print("Methodology: Uses portfolio sensitivities (Greeks) and covariance matrix")
    print("Advantages: Fast, scalable, excellent attribution, regulatory standard")
    print("Best For: Large portfolios, linear products, real-time monitoring")
    print()

    portfolio = create_sample_portfolio()
    historical_data = generate_historical_data()

    print("=" * 80)
    print("Portfolio Summary")
    print("=" * 80)
    print(f"Portfolio Name: {portfolio.portfolio_name}")
    print(f"Number of positions: {len(portfolio.positions)}")
    print(f"Portfolio value: ${portfolio.get_portfolio_value():,.2f}")
    print()

    for i, (pos_id, position) in enumerate(portfolio.positions.items(), 1):
        product_name = position.product.__class__.__name__
        strike = position.product.strike
        expiry = position.product.exercise_date.strftime("%Y-%m-%d")
        print(
            f"  {i}. {product_name:25s} | Strike: ${strike:6.1f} | Expiry: {expiry} | Qty: {position.quantity:4d}"
        )

    print()

    # Configure equity risk factors
    equity_factors = EquityRiskFactorConfig(
        include_spot=True, include_vol=True, include_rate=True, include_div_yield=False
    )

    print("=" * 80)
    print("Parametric VaR Calculation")
    print("=" * 80)

    # Configure VaR with full attribution
    parametric_config = VaRConfig(
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

    parametric_engine = ParametricVaREngine(config=parametric_config)
    parametric_result = parametric_engine.calculate_var(portfolio, historical_data)

    # Create report generator instance
    report_generator = VaRReportGenerator()

    # Display summary results
    summary = report_generator.generate_summary(parametric_result)
    print(summary)

    print()
    print("=" * 80)
    print("Component VaR Attribution (Position-Level)")
    print("=" * 80)
    print()
    print(
        "Component VaR shows how much each position contributes to total portfolio VaR."
    )
    print("Sum of Component VaRs = Total Portfolio VaR (Euler decomposition)")
    print()

    if parametric_result.component_var:
        # Create attribution table
        attribution_data = []
        total_component_var = 0
        for pos_id, comp_var in parametric_result.component_var.items():
            position = portfolio.positions[pos_id]
            product_name = position.product.__class__.__name__
            attribution_data.append(
                {
                    "Position ID": pos_id,
                    "Product": product_name,
                    "Quantity": position.quantity,
                    "Component VaR": comp_var,
                    "% of Total VaR": (
                        (comp_var / parametric_result.var * 100)
                        if parametric_result.var > 0
                        else 0
                    ),
                }
            )
            total_component_var += comp_var

        df = pd.DataFrame(attribution_data)
        df = df.sort_values("Component VaR", ascending=False)

        for _, row in df.iterrows():
            print(
                f"{row['Position ID']:10s} | {row['Product']:25s} | "
                f"Qty: {row['Quantity']:4d} | "
                f"Comp VaR: ${row['Component VaR']:8,.2f} | "
                f"{row['% of Total VaR']:5.1f}%"
            )

        print()
        print(f"Sum of Component VaRs: ${total_component_var:,.2f}")
        print(f"Total Portfolio VaR:   ${parametric_result.var:,.2f}")
        print(
            f"Reconciliation: {'✓' if abs(total_component_var - parametric_result.var) < 1e-6 else '✗'}"
        )

    print()
    print("=" * 80)
    print("Factor VaR Attribution (Risk Factor-Level)")
    print("=" * 80)
    print()
    print(
        "Factor VaR shows risk contribution by market factor (spot, vol, rate, dividend)."
    )
    print()

    if parametric_result.factor_var:
        factor_report = report_generator.generate_factor_report(parametric_result)
        print(factor_report)

    print()
    print("=" * 80)
    print("Marginal and Incremental VaR")
    print("=" * 80)
    print()
    print("Marginal VaR: ∂VaR/∂x_i (change in VaR from adding $1 of position)")
    print("Incremental VaR: Full portfolio VaR with/without position")
    print()

    if parametric_result.marginal_var:
        print("Top 5 Positions by Marginal VaR:")
        marginal_sorted = sorted(
            [(pos_id, mvar) for pos_id, mvar in parametric_result.marginal_var.items()],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]

        for pos_id, mvar in marginal_sorted:
            print(f"  {pos_id:10s}: ${mvar:8,.2f}")

    print()

    if parametric_result.incremental_var:
        print("Top 5 Positions by Incremental VaR:")
        incremental_sorted = sorted(
            [
                (pos_id, ivar)
                for pos_id, ivar in parametric_result.incremental_var.items()
            ],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]

        for pos_id, ivar in incremental_sorted:
            print(f"  {pos_id:10s}: ${ivar:8,.2f}")

    print()
    print("=" * 80)
    print("Stressed VaR (SVaR)")
    print("=" * 80)
    print()
    print("SVaR calculates VaR over a stressed (crisis) period.")
    print("Basel requirement: 12-month period with highest VaR in last 5 years.")
    print()

    if parametric_result.stressed_var:
        print(f"Regular VaR: ${parametric_result.var:,.2f}")
        print(f"Stressed VaR: ${parametric_result.stressed_var:,.2f}")
        print(
            f"SVaR / VaR Ratio: {parametric_result.stressed_var / parametric_result.var:.2f}x"
        )
    else:
        print("SVaR not calculated in this demo (would require 5 years of data)")

    print()
    print("=" * 80)
    print("When to Use Parametric VaR")
    print("=" * 80)
    print()
    print("✓ RECOMMENDED FOR:")
    print("  • Large equity portfolios (> 1,000 positions)")
    print("  • Linear risk exposures (stocks, forwards, swaps)")
    print("  • Real-time risk monitoring")
    print("  • Regulatory reporting (Basel III/IV)")
    print("  • When Greeks are reliable (ATM options, stable volatility)")
    print()
    print("✗ NOT RECOMMENDED FOR:")
    print("  • Highly non-linear products (deep OTM options)")
    print("  • Portfolios with fat-tail risk")
    print("  • Path-dependent derivatives (Asian, barrier options)")
    print("  • When volatility is stochastic or jumps")
    print()
    print("=" * 80)
    print("Demonstration Complete")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print(
        f"  • VaR: ${parametric_result.var:,.2f} ({parametric_result.var_as_pct*100:.2f}% of portfolio)"
    )
    print(
        f"  • CVaR: ${parametric_result.cvar:,.2f} ({parametric_result.cvar/portfolio.get_portfolio_value()*100:.2f}% of portfolio)"
    )
    print(f"  • Execution time: {parametric_result.execution_time_seconds:.4f} seconds")
    print("  • Excellent attribution for risk decomposition")
    print("  • Fast and scalable for large portfolios")
    print()


if __name__ == "__main__":
    main()
