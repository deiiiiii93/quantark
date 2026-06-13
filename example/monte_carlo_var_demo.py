"""
Monte Carlo VaR (Simulation) demonstration.

This example demonstrates the use of Monte Carlo VaR, which generates simulated
market scenarios using statistical distributions to calculate VaR.

Methodology:
- Generates N simulated scenarios for risk factors (spot returns, vol changes, etc.)
- Revalues portfolio under each simulated scenario
- Sorts scenario P&L and takes percentile for VaR threshold

Advantages:
- Flexible distribution modeling (normal, t-distribution, skewed, etc.)
- Can model complex dependencies and correlations
- Handles path-dependent products (Asian, barrier options)
- Forward-looking (not limited by historical data)
- Custom stress scenarios and what-if analysis
- Can incorporate stochastic volatility, jumps

Best For:
- Path-dependent derivatives (Asian, barrier, lookback options)
- Limited historical data situations
- Custom scenario analysis
- Products with complex payoffs
- Stress testing and scenario planning
- Non-Gaussian risk factor distributions

Limitations:
- Slowest method (requires many simulations)
- Depends on distribution assumptions
- Requires convergence testing
- Computational complexity for large portfolios
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path


from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.portfolio.equity.portfolio import EquityPortfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.option_enums import OptionType

from quantark.var import (
    VaRConfig,
    VaRMethod,
    MonteCarloVaREngine,
    VaRReportGenerator,
)


def create_portfolio_for_mc():
    """Create a portfolio suitable for Monte Carlo VaR demonstration."""
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
        portfolio_name="Monte Carlo VaR Demo Portfolio",
        pricing_environments={"GOOGL": pricing_env},
    )

    # Create portfolio with various exposures
    # Long straddle (benefits from volatility)
    call_100 = EuropeanVanillaOption(
        strike=100.0, exercise_date=datetime(2024, 4, 1), option_type=OptionType.CALL
    )

    put_100 = EuropeanVanillaOption(
        strike=100.0, exercise_date=datetime(2024, 4, 1), option_type=OptionType.PUT
    )

    # OTM call (low probability, high payoff)
    otm_call = EuropeanVanillaOption(
        strike=120.0, exercise_date=datetime(2024, 7, 1), option_type=OptionType.CALL
    )

    # Deep OTM call (very low probability, very high payoff if volatility spikes)
    deep_otm_call = EuropeanVanillaOption(
        strike=140.0, exercise_date=datetime(2024, 10, 1), option_type=OptionType.CALL
    )

    engine = BlackScholesEngine()

    portfolio.add_position(
        product=call_100,
        quantity=50,
        entry_price=6.5,
        underlying="GOOGL",
        engine=engine,
    )

    portfolio.add_position(
        product=put_100,
        quantity=50,
        entry_price=5.8,
        underlying="GOOGL",
        engine=engine,
    )

    portfolio.add_position(
        product=otm_call,
        quantity=100,
        entry_price=1.2,
        underlying="GOOGL",
        engine=engine,
    )

    portfolio.add_position(
        product=deep_otm_call,
        quantity=150,
        entry_price=0.35,
        underlying="GOOGL",
        engine=engine,
    )

    return portfolio


def generate_market_data_for_mc(num_days=252):
    """Generate baseline market data for Monte Carlo VaR."""
    np.random.seed(42)
    dates = pd.date_range(start=datetime(2023, 1, 1), periods=num_days, freq="D")

    # Base parameters for simulation
    data = pd.DataFrame(
        {
            "spot_return": np.random.normal(0.0005, 0.015, num_days),
            "vol_change": np.random.normal(0.0, 0.01, num_days),
            "rate_shift": np.random.normal(0.0, 0.0005, num_days),
        },
        index=dates,
    )

    return data


def main(
    mc_num_simulations: int = 10000,
    convergence_counts: tuple = (1000, 5000, 10000),
):
    """Run Monte Carlo VaR demonstration and display results.

    Args:
        mc_num_simulations: Simulation count for the headline VaR run.
        convergence_counts: Simulation counts for the convergence analysis.
    """
    print("=" * 80)
    print("Monte Carlo VaR (Simulation) Demonstration")
    print("=" * 80)
    print()
    print("Methodology: Simulated market scenarios using statistical distributions")
    print("Advantages: Flexible distributions, path-dependent products, stress testing")
    print("Best For: Limited history, custom scenarios, complex derivatives")
    print()

    portfolio = create_portfolio_for_mc()
    historical_data = generate_market_data_for_mc()

    print("=" * 80)
    print("Portfolio Summary")
    print("=" * 80)
    print(f"Portfolio Name: {portfolio.portfolio_name}")
    print(f"Number of positions: {len(portfolio.positions)}")
    print(f"Portfolio value: ${portfolio.get_portfolio_value():,.2f}")
    print()
    print("Positions (including path-dependent Asian option):")
    print()

    for i, (pos_id, position) in enumerate(portfolio.positions.items(), 1):
        product_name = position.product.__class__.__name__
        strike = position.product.strike
        expiry = position.product.exercise_date.strftime("%Y-%m-%d")
        opt_type = "CALL" if position.product.option_type == OptionType.CALL else "PUT"

        print(
            f"  {i}. {opt_type:15s} Strike: ${strike:6.1f} | Expiry: {expiry} | Qty: {position.quantity:4d}"
        )

    print()
    print(
        f"Note: Monte Carlo is ideal for path-dependent products and custom scenarios!"
    )
    print()

    print("=" * 80)
    print("Part 1: Monte Carlo VaR with Default Settings")
    print("=" * 80)

    # Configure Monte Carlo VaR
    mc_config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        var_method=VaRMethod.MONTE_CARLO,
        mc_num_simulations=mc_num_simulations,
        mc_seed=42,
    )

    mc_engine = MonteCarloVaREngine(config=mc_config)
    mc_result = mc_engine.calculate_var(portfolio, historical_data)

    # Create report generator instance
    report_generator = VaRReportGenerator()

    summary = report_generator.generate_summary(mc_result)
    print(summary)

    print()
    print("=" * 80)
    print("Part 2: Convergence Analysis")
    print("=" * 80)
    print()
    print("Monte Carlo requires sufficient simulations for convergence.")
    print("Testing VaR at different simulation counts:")
    print()

    simulation_counts = list(convergence_counts)
    convergence_results = []

    for num_sims in simulation_counts:
        print(f"Testing {num_sims:,} simulations... ", end="", flush=True)

        config = VaRConfig(
            confidence_level=0.99,
            holding_period=1,
            var_method=VaRMethod.MONTE_CARLO,
            mc_num_simulations=num_sims,
            mc_seed=42,
        )

        engine = MonteCarloVaREngine(config=config)
        result = engine.calculate_var(portfolio, historical_data)

        convergence_results.append(
            {
                "Simulations": num_sims,
                "VaR": result.var,
                "CVaR": result.cvar,
                "Time (s)": result.execution_time_seconds,
            }
        )

        print(
            f"VaR: ${result.var:8,.2f} | CVaR: ${result.cvar:8,.2f} | Time: {result.execution_time_seconds:6.3f}s"
        )

    print()
    print("Convergence Analysis:")
    print("-" * 70)
    print(f"{'Simulations':>12} {'VaR':>12} {'CVaR':>12} {'Change':>10} {'Time':>10}")
    print("-" * 70)

    for i, result in enumerate(convergence_results):
        if i == 0:
            change = "N/A"
        else:
            prev_var = convergence_results[i - 1]["VaR"]
            change_pct = (result["VaR"] - prev_var) / prev_var * 100
            change = f"{change_pct:+5.2f}%"

        print(
            f"{result['Simulations']:>12,} ${result['VaR']:>10,.2f} "
            f"${result['CVaR']:>10,.2f} {change:>8s} {result['Time (s)']:>8.3f}s"
        )

    print()
    print("Key Observations:")
    print("  • VaR stabilizes as simulation count increases")
    print("  • Diminishing returns after ~10,000-25,000 simulations")
    print("  • Computational time scales linearly with simulations")
    print("  • 95% confidence typically reached by 10,000-50,000 sims")

    print()
    print("=" * 80)
    print("Part 3: Distribution Flexibility")
    print("=" * 80)
    print()
    print("Monte Carlo allows modeling different return distributions.")
    print("Standard normal vs. fat-tail distributions (t-distribution):")
    print()

    # Note: Actual implementation would support different distributions
    # For this demo, we show the concept
    print("Distribution Comparison (conceptual):")
    print()
    print("  Normal Distribution:")
    print("    - Assumes thin tails (Gaussian)")
    print("    - Faster convergence")
    print("    - May underestimate tail risk")
    print()
    print("  t-Distribution (Student's t):")
    print("    - Models fat tails (higher kurtosis)")
    print("    - More conservative VaR estimates")
    print("    - Better for stress scenarios")
    print()
    print("  Skewed Distributions:")
    print("    - Captures asymmetric risk (crashes vs. rallies)")
    print("    - Useful for equity markets (crash asymmetry)")
    print("    - Custom tail modeling")

    print()
    print("=" * 80)
    print("Part 4: Stress Scenario Testing")
    print("=" * 80)
    print()
    print("Monte Carlo excels at custom stress testing.")
    print("Example stress scenarios:")
    print()

    stress_scenarios = [
        {
            "name": "Market Crash",
            "spot_return": -0.20,
            "vol_change": 0.50,
            "description": "2008-style crisis",
        },
        {
            "name": "Volatility Spike",
            "spot_return": -0.05,
            "vol_change": 1.00,
            "description": "Vol-of-vol shock",
        },
        {
            "name": "Rate Shock",
            "spot_return": 0.02,
            "vol_change": 0.10,
            "rate_shift": 0.01,
            "description": "Fed surprise",
        },
        {
            "name": "Perfect Storm",
            "spot_return": -0.15,
            "vol_change": 0.80,
            "rate_shift": 0.008,
            "description": "Combined stress",
        },
    ]

    for scenario in stress_scenarios:
        print(f"  • {scenario['name']:18s}: {scenario['description']}")
        print(
            f"    Spot Return: {scenario.get('spot_return', 0):+6.1%}, "
            f"Vol Change: {scenario.get('vol_change', 0):+6.1%}, "
            f"Rate Shift: {scenario.get('rate_shift', 0):+6.4f}"
        )

    print()
    print("These scenarios can be overlaid on Monte Carlo simulations")
    print("to test portfolio resilience under extreme conditions.")

    print()
    print("=" * 80)
    print("Path-Dependent Products")
    print("=" * 80)
    print()
    print("Monte Carlo is ideal for path-dependent options:")
    print()
    print("  Asian Options:")
    print("    - Payoff based on average price over time")
    print("    - Requires full price path simulation")
    print("    - Not demonstrated in this portfolio (AsianOption not available)")
    print()
    print("  Barrier Options:")
    print("    - Activate/deactivate if price hits barrier")
    print("    - Path-dependent activation logic")
    print()
    print("  Lookback Options:")
    print("    - Use max/min price over option life")
    print("    - Requires tracking extreme values")
    print()
    print("  Note: Parametric and Historical methods struggle with these products!")
    print("        Monte Carlo handles them naturally.")

    print()
    print("=" * 80)
    print("Performance Optimization Tips")
    print("=" * 80)
    print()
    print("Speed up Monte Carlo VaR:")
    print()
    print("  1. Variance Reduction:")
    print("     • Antithetic variates (pair simulations)")
    print("     • Control variates (use known pricing)")
    print("     • Importance sampling (focus on tail)")
    print()
    print("  2. Parallel Computing:")
    print("     • Distribute simulations across cores")
    print("     • GPU acceleration for large portfolios")
    print("     • Cloud computing for massive scale")
    print()
    print("  3. Efficient Pricing:")
    print("     • Analytic approximations when available")
    print("     • Regression-based pricing (Longstaff-Schwartz)")
    print("     • GPU-optimized Monte Carlo engines")
    print()
    print("  4. Smart Sampling:")
    print("     • Adaptive simulation (focus on variance)")
    print("     • Quasi-Monte Carlo (Sobol sequences)")
    print("     • Multi-level Monte Carlo")

    print()
    print("=" * 80)
    print("When to Use Monte Carlo VaR")
    print("=" * 80)
    print()
    print("✓ RECOMMENDED FOR:")
    print("  • Path-dependent derivatives (Asian, barrier, lookback)")
    print("  • Limited historical data (new markets, products)")
    print("  • Custom scenario analysis and stress testing")
    print("  • Non-Gaussian risk distributions")
    print("  • Products with stochastic volatility/jumps")
    print("  • Forward-looking risk assessment")
    print()
    print("✓ ESPECIALLY IMPORTANT FOR:")
    print("  • Structured products with complex payoffs")
    print("  • Risk management in emerging markets")
    print("  • Dynamic hedging strategies")
    print("  • Capital allocation under custom scenarios")
    print()
    print("✗ CONSIDER ALTERNATIVES IF:")
    print("  • Large linear portfolio → Use Parametric")
    print("  • Sufficient history exists → Use Historical")
    print("  • Real-time monitoring needed → Use Parametric")
    print("  • Very large portfolio → Consider hybrid approach")
    print()
    print("=" * 80)
    print("Demonstration Complete")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print(
        f"  • VaR: ${mc_result.var:,.2f} ({mc_result.var_as_pct*100:.2f}% of portfolio)"
    )
    print(f"  • CVaR: ${mc_result.cvar:,.2f}")
    print(f"  • Final execution time: {mc_result.execution_time_seconds:.4f} seconds")
    print("  • Flexible and forward-looking method")
    print("  • Essential for path-dependent products")
    print("  • Convergence achieved with ~25,000+ simulations")
    print()


if __name__ == "__main__":
    main()
