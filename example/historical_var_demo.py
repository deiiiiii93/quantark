"""
Historical VaR (Full Revaluation) demonstration.

This example demonstrates the use of Historical VaR, which revaluates the entire portfolio
under actual historical market scenarios.

Methodology:
- Uses historical market data (returns, volatility changes, rate shifts)
- Revalues portfolio under each historical scenario
- Sorts scenario P&L and takes percentile for VaR threshold

Advantages:
- Most accurate method for options and non-linear products
- Captures actual historical correlation structure
- No distributional assumptions
- Handles fat tails and market stress naturally
- Full revaluation accounts for gamma and vega effects

Best For:
- Option portfolios (European, American, exotic)
- Non-linear payoffs and derivatives
- Portfolios with significant convexity
- High accuracy requirements
- Capturing actual market stress

Limitations:
- Slower than parametric (requires full revaluation)
- Limited by available historical data
- May not capture future risks not in history
- Assumes historical patterns repeat
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
    HistoricalVaREngine,
    VaRReportGenerator,
)


def create_options_portfolio():
    """Create an options portfolio optimized to show non-linear effects."""
    valuation_date = datetime(2024, 1, 1)

    spot_quote = SpotQuote(spot=100.0, timestamp=valuation_date)
    vol_surface = FlatVolSurface(volatility=0.30)
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
        portfolio_name="Historical VaR Demo - Options Portfolio",
        pricing_environments={"MSFT": pricing_env},
    )

    # Create portfolio with significant non-linear exposure
    # ATM options (high gamma, high vega)
    atm_call = EuropeanVanillaOption(
        strike=100.0, exercise_date=datetime(2024, 3, 1), option_type=OptionType.CALL
    )

    atm_put = EuropeanVanillaOption(
        strike=100.0, exercise_date=datetime(2024, 3, 1), option_type=OptionType.PUT
    )

    # Slightly OTM call (positive gamma, benefits from volatility)
    otm_call = EuropeanVanillaOption(
        strike=110.0, exercise_date=datetime(2024, 6, 1), option_type=OptionType.CALL
    )

    # Deep OTM call (low probability, high payoff if volatility spikes)
    deep_otm_call = EuropeanVanillaOption(
        strike=130.0, exercise_date=datetime(2024, 9, 1), option_type=OptionType.CALL
    )

    engine = BlackScholesEngine()

    portfolio.add_position(
        product=atm_call,
        quantity=100,
        entry_price=5.2,
        underlying="MSFT",
        engine=engine,
    )

    portfolio.add_position(
        product=atm_put,
        quantity=100,
        entry_price=4.8,
        underlying="MSFT",
        engine=engine,
    )

    portfolio.add_position(
        product=otm_call,
        quantity=150,
        entry_price=2.1,
        underlying="MSFT",
        engine=engine,
    )

    portfolio.add_position(
        product=deep_otm_call,
        quantity=200,
        entry_price=0.75,
        underlying="MSFT",
        engine=engine,
    )

    return portfolio


def generate_historical_data(num_days=500):
    """Generate synthetic historical market data with fat tails."""
    np.random.seed(42)
    dates = pd.date_range(start=datetime(2022, 1, 1), periods=num_days, freq="D")

    # Generate returns with occasional extreme events (fat tails)
    spot_returns = np.random.normal(0.0003, 0.018, num_days)

    # Add some extreme events (5% probability)
    extreme_events = np.random.choice([0, 1], size=num_days, p=[0.95, 0.05])
    extreme_shocks = np.random.normal(0, 0.08, num_days) * extreme_events
    spot_returns += extreme_shocks

    # Volatility clustering: high vol periods
    vol_changes = np.random.normal(0.0, 0.015, num_days)
    vol_clustering = np.abs(np.random.normal(0, 0.02, num_days)) * extreme_events
    vol_changes += vol_clustering

    # Interest rates with mean reversion
    rate_shifts = np.random.normal(0.0, 0.0008, num_days)

    data = pd.DataFrame(
        {
            "spot_return": spot_returns,
            "vol_change": vol_changes,
            "rate_shift": rate_shifts,
        },
        index=dates,
    )

    return data


def main():
    """Run Historical VaR demonstration and display results."""
    print("=" * 80)
    print("Historical VaR (Full Revaluation) Demonstration")
    print("=" * 80)
    print()
    print("Methodology: Full portfolio revaluation under historical market scenarios")
    print("Advantages: Most accurate for options, captures non-linear effects")
    print("Best For: Option portfolios, derivatives, high accuracy requirements")
    print()

    portfolio = create_options_portfolio()
    historical_data = generate_historical_data()

    print("=" * 80)
    print("Portfolio Summary")
    print("=" * 80)
    print(f"Portfolio Name: {portfolio.portfolio_name}")
    print(f"Number of positions: {len(portfolio.positions)}")
    print(f"Portfolio value: ${portfolio.get_portfolio_value():,.2f}")
    print()
    print("Option Positions (high gamma/vega exposure):")
    print()

    for i, (pos_id, position) in enumerate(portfolio.positions.items(), 1):
        strike = position.product.strike
        expiry = position.product.exercise_date.strftime("%Y-%m-%d")
        opt_type = "CALL" if position.product.option_type == OptionType.CALL else "PUT"
        moneyness = "ATM" if abs(strike - 100) < 5 else "OTM" if strike > 100 else "ITM"
        print(
            f"  {i}. {opt_type:4s} {strike:6.1f} ({moneyness:3s}) | Expiry: {expiry} | Qty: {position.quantity:4d}"
        )

    print()
    print(
        f"Historical data: {len(historical_data)} days from {historical_data.index[0].date()} to {historical_data.index[-1].date()}"
    )
    print()

    print("=" * 80)
    print("Historical VaR Calculation")
    print("=" * 80)

    # Configure VaR
    historical_config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        lookback_days=min(252, len(historical_data)),
        var_method=VaRMethod.HISTORICAL,
    )

    historical_engine = HistoricalVaREngine(config=historical_config)
    historical_result = historical_engine.calculate_var(portfolio, historical_data)

    # Create report generator instance
    report_generator = VaRReportGenerator()

    # Display summary results
    summary = report_generator.generate_summary(historical_result)
    print(summary)

    print()
    print("=" * 80)
    print("Scenario Analysis - Worst Scenarios")
    print("=" * 80)
    print()
    print("Historical VaR ranks all scenarios by portfolio P&L.")
    print("Worst scenarios show actual historical events that caused maximum losses.")
    print()

    num_scenarios = min(15, len(historical_result.scenarios))
    print(f"Top {num_scenarios} worst scenarios:")
    print()

    # Display worst scenarios
    if (
        historical_result.scenarios is not None
        and not historical_result.scenarios.empty
    ):
        # Check what columns are available
        columns = historical_result.scenarios.columns.tolist()
        print(f"Available columns: {columns}")

        # Find the P&L column (could be 'portfolio_pnl' or similar)
        pnl_col = None
        for col in columns:
            if "pnl" in col.lower() or "loss" in col.lower():
                pnl_col = col
                break

        if pnl_col:
            worst_scenarios = historical_result.scenarios.nsmallest(
                num_scenarios, pnl_col
            )

            for i, (_, scenario) in enumerate(worst_scenarios.iterrows(), 1):
                scenario_date = scenario.get("date", "N/A")
                pnl = scenario.get(pnl_col, 0)
                spot_ret = scenario.get("spot_return", 0)
                vol_chg = scenario.get("vol_change", 0)

                print(
                    f"  {i:2d}. Date: {scenario_date} | P&L: ${pnl:8,.2f} | "
                    f"Spot Return: {spot_ret:6.2%} | Vol Change: {vol_chg:6.3f}"
                )
        else:
            print(f"  No P&L column found in scenarios. Available columns: {columns}")

    else:
        print("  No scenarios available")

    print()
    print("=" * 80)
    print("Scenario Distribution Statistics")
    print("=" * 80)
    print()

    if (
        historical_result.scenarios is not None
        and not historical_result.scenarios.empty
    ):
        # Find P&L column
        pnl_col = None
        for col in historical_result.scenarios.columns:
            if "pnl" in col.lower() or "loss" in col.lower():
                pnl_col = col
                break

        if pnl_col:
            pnl_values = historical_result.scenarios[pnl_col]

            stats = {
                "Mean": pnl_values.mean(),
                "Std Dev": pnl_values.std(),
                "Skewness": pnl_values.skew(),
                "Kurtosis": pnl_values.kurtosis(),
                "Min": pnl_values.min(),
                "1st Percentile": pnl_values.quantile(0.01),
                "5th Percentile": pnl_values.quantile(0.05),
                "50th Percentile (Median)": pnl_values.median(),
                "95th Percentile": pnl_values.quantile(0.95),
                "99th Percentile": pnl_values.quantile(0.99),
                "Max": pnl_values.max(),
            }

            for key, value in stats.items():
                if "Percentile" in key or key in ["Mean", "Std Dev"]:
                    print(f"{key:25s}: ${value:10,.2f}")
                elif key in ["Skewness", "Kurtosis"]:
                    print(f"{key:25s}: {value:10.3f}")
                else:
                    print(f"{key:25s}: ${value:10,.2f}")
        else:
            print("Could not find P&L column in scenarios for statistics")

    print()
    print("=" * 80)
    print("Non-Linear Effects Demonstration")
    print("=" * 80)
    print()
    print("Historical VaR correctly captures non-linear effects (gamma, vega) because")
    print("it revalues the portfolio under each scenario using the full pricing model.")
    print()
    print("Key Observations:")
    print(f"  • VaR ({historical_result.var:,.2f}) reflects actual option behavior")
    print(f"  • CVaR ({historical_result.cvar:,.2f}) shows tail risk beyond VaR")
    print("  • Fat tails captured through extreme scenarios")
    print("  • Volatility changes affect option values (vega)")
    print("  • Large moves create gamma effects (convexity)")
    print()

    print("=" * 80)
    print("Comparison: Why Historical vs Parametric?")
    print("=" * 80)
    print()

    # Calculate parametric VaR for comparison
    from var import ParametricVaREngine, EquityRiskFactorConfig

    parametric_config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        lookback_days=min(252, len(historical_data)),
        var_method=VaRMethod.PARAMETRIC,
        equity_factors=EquityRiskFactorConfig(
            include_spot=True, include_vol=True, include_rate=True
        ),
    )

    parametric_engine = ParametricVaREngine(config=parametric_config)
    parametric_result = parametric_engine.calculate_var(portfolio, historical_data)

    print(f"{'Method':<20} {'VaR':>12} {'CVaR':>12} {'Time (s)':>10}")
    print("-" * 60)
    print(
        f"{'Parametric':<20} ${parametric_result.var:>10,.2f} ${parametric_result.cvar:>10,.2f} {parametric_result.execution_time_seconds:>8.4f}"
    )
    print(
        f"{'Historical':<20} ${historical_result.var:>10,.2f} ${historical_result.cvar:>10,.2f} {historical_result.execution_time_seconds:>8.4f}"
    )
    print()
    print(
        f"Historical vs Parametric VaR Ratio: {historical_result.var / parametric_result.var:.2f}x"
    )
    print()

    if historical_result.var > parametric_result.var * 1.1:
        print(
            "Note: Historical VaR is higher than parametric, indicating fat-tail effects"
        )
        print(
            "      or non-linear risks not captured by normal distribution assumption."
        )
    elif parametric_result.var > historical_result.var * 1.1:
        print(
            "Note: Parametric VaR is higher, possibly due to conservative covariance estimates."
        )
    else:
        print(
            "Note: VaR methods show similar results, suggesting linear-ish portfolio behavior."
        )

    print()
    print("=" * 80)
    print("When to Use Historical VaR")
    print("=" * 80)
    print()
    print("✓ RECOMMENDED FOR:")
    print("  • Option portfolios (all types: vanilla, exotic, path-dependent)")
    print("  • Non-linear payoffs and derivatives")
    print("  • High accuracy requirements")
    print("  • Capturing market crashes and stress (fat tails)")
    print("  • When Greeks may be unreliable (deep OTM, high volatility)")
    print("  • Portfolios with significant convexity")
    print()
    print("✓ ESPECIALLY IMPORTANT FOR:")
    print("  • Gamma trading strategies")
    print("  • Volatility products (VIX, variance swaps)")
    print("  • Barrier and knockout options")
    print("  • High-frequency rebalancing")
    print()
    print("✗ LIMITATIONS:")
    print("  • Slower than parametric (full revaluation required)")
    print("  • Limited by available history (need 3-5 years minimum)")
    print("  • Assumes past patterns predict future risk")
    print("  • May miss new types of market stress")
    print()
    print("=" * 80)
    print("Demonstration Complete")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print(
        f"  • VaR: ${historical_result.var:,.2f} ({historical_result.var_as_pct*100:.2f}% of portfolio)"
    )
    print(f"  • CVaR: ${historical_result.cvar:,.2f}")
    print(f"  • Execution time: {historical_result.execution_time_seconds:.4f} seconds")
    print("  • Most accurate for options and non-linear products")
    print("  • Captures actual historical market behavior")
    print(f"  • {len(historical_data)} scenarios analyzed")
    print()


if __name__ == "__main__":
    main()
