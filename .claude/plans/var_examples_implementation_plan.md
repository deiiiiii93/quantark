# VaR Example Scripts Implementation Plan

## Overview

This document outlines the implementation plan for creating 4 focused Value-at-Risk (VaR) example scripts based on the existing `portfolio_var_demo.py` pattern. Each script will demonstrate a specific VaR methodology with detailed explanations, practical use cases, and comprehensive result reporting.

## Project Structure

```
/Users/fuxinyao/quant-ark/example/
├── parametric_var_demo.py          [NEW] Variance-covariance method demo
├── historical_var_demo.py          [NEW] Historical simulation method demo
├── monte_carlo_var_demo.py         [NEW] Monte Carlo simulation demo
├── var_backtest_demo.py            [NEW] VaR model validation demo
└── portfolio_var_demo.py           [EXISTING] Combined demo (reference pattern)
```

## Common Pattern and Conventions

All example scripts will follow the established pattern from `portfolio_var_demo.py`:

### File Structure Pattern

Each script will have:
1. **Module docstring** - Description of the method and use cases
2. **Import statements** - All necessary imports from QuantArk modules
3. **Helper functions**:
   - `create_sample_portfolio()` - Portfolio setup
   - `generate_historical_data(num_days=300)` - Market data generation
4. **Main demonstration**:
   - Portfolio creation and display
   - VaR configuration
   - VaR calculation
   - Comprehensive result reporting
   - Performance metrics
5. **Educational comments** - Explaining when and why to use each method

### Import Pattern

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from param import (ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote)
from portfolio.equity.portfolio import EquityPortfolio
from priceenv import PricingEnvironment
from util.enum.option_enums import OptionType

# VaR-specific imports
from var import (VaRConfig, VaRMethod, EquityRiskFactorConfig,
                 ParametricVaREngine, HistoricalVaREngine, MonteCarloVaREngine,
                 VaRReportGenerator, ComponentVaRCalculator, VaRAttributor,
                 VaRBacktester, VaRBacktestResult)
```

### Portfolio Creation Pattern

```python
def create_sample_portfolio():
    """Create a sample equity options portfolio with specific characteristics."""
    valuation_date = datetime(2024, 1, 1)

    # Market data setup
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

    # Add options positions
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
```

### Historical Data Generation Pattern

```python
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
```

## Implementation Details

### 1. Parametric VaR Demo (`parametric_var_demo.py`)

**Purpose**: Demonstrate the variance-covariance (sensitivity-based) method with component and factor attribution.

**Key Features to Highlight**:
- Fast calculation using sensitivities (Greeks)
- Component VaR and Euler decomposition
- Factor VaR attribution (spot, vol, rate)
- Best for: Large portfolios, real-time monitoring, linear products
- Worst for: Non-linear products (but supports delta-gamma approximation)

**Detailed Structure**:

```python
"""
Parametric VaR (Variance-Covariance) Demonstration.

This example demonstrates:
1. Parametric VaR calculation using portfolio sensitivities
2. Component VaR with Euler decomposition
3. Factor VaR attribution by risk factor
4. Speed and scalability advantages
5. When to use parametric vs other methods

Parametric VaR (Variance-Covariance Method):
- Uses portfolio sensitivities (delta, gamma, vega, rho, DV01)
- Leverages historical covariance matrix of risk factors
- Fastest calculation method (closed-form solutions)
- Industry standard for equity and FI trading
- Well-suited for linear portfolios
- Can handle non-linear with delta-gamma approximation

Advantages:
+ Fastest calculation (scalable to very large portfolios)
+ Closed-form Greeks support
+ Real-time risk monitoring
+ Efficient for backtesting
+ Supports attribution (component, factor, marginal, incremental)

Disadvantages:
- Assumes linear relationship (or approximations for non-linear)
- Distributional assumptions (normally distributed returns)
- Limited accuracy for options and derivatives
- Requires reliable Greeks calculations
- May not capture fat tails

Mathematical Foundation:
VaR = z_score * sqrt(s^T * Σ * s)
where:
- s = sensitivity vector (Greeks/DV01)
- Σ = covariance matrix of risk factors
- z_score = inverse CDF of normal distribution at confidence level
"""

import time

def main():
    print("=" * 80)
    print("PARAMETRIC VaR (VARIANCE-COVARIANCE) DEMONSTRATION")
    print("=" * 80)

    # Create portfolio
    portfolio = create_sample_portfolio()
    historical_data = generate_historical_data()

    print(f"\nPortfolio: {portfolio.portfolio_name}")
    print(f"Number of positions: {len(portfolio.positions)}")
    print(f"Portfolio value: ${portfolio.get_portfolio_value():,.2f}")

    # Configure risk factors (equity)
    equity_factors = EquityRiskFactorConfig(
        include_spot=True,      # Delta sensitivity
        include_vol=True,       # Vega sensitivity
        include_rate=True,      # Rho sensitivity
        include_div_yield=False  # No dividend yield shift
    )

    # Configure VaR calculation
    config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        lookback_days=252,  # 1 year of data
        var_method=VaRMethod.PARAMETRIC,
        equity_factors=equity_factors,
        calculate_factor_var=True,      # Enable factor attribution
        calculate_component_var=True,   # Enable component VaR
        calculate_marginal_var=True,    # Enable marginal VaR
        calculate_incremental_var=True, # Enable incremental VaR
        calculate_stressed_var=True,    # Enable stressed VaR
    )

    # Calculate VaR
    print("\n" + "=" * 80)
    print("CALCULATING PARAMETRIC VaR")
    print("=" * 80)

    start_time = time.time()

    engine = ParametricVaREngine(config=config)
    result = engine.calculate_var(portfolio, historical_data)

    calc_time = time.time() - start_time

    # Display results
    print(f"\nCalculation completed in {calc_time:.3f} seconds")

    # Summary report
    summary = VaRReportGenerator.generate_summary(result)
    for key, value in summary.items():
        print(f"{key:30s}: {value}")

    # Component VaR report
    if result.component_var:
        print("\n" + "=" * 80)
        print("COMPONENT VaR (POSITION-LEVEL ATTRIBUTION)")
        print("=" * 80)
        print(f"{'Position ID':<20} {'Component VaR':>15} {'% of Total':>15}")
        print("-" * 80)
        sorted_components = sorted(
            result.component_var.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for pos_id, comp_var in sorted_components:
            pct = (comp_var / result.var) * 100
            print(f"{pos_id:<20} ${comp_var:>13,.2f} {pct:>13.2f}%")

    # Factor VaR attribution
    if result.factor_var:
        print("\n" + "=" * 80)
        print("FACTOR VaR ATTRIBUTION")
        print("=" * 80)
        print(f"{'Risk Factor':<20} {'Factor VaR':>15} {'% of Total':>15}")
        print("-" * 80)
        sorted_factors = sorted(
            result.factor_var.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for factor, factor_var in sorted_factors:
            pct = (factor_var / result.var) * 100
            print(f"{factor:<20} ${factor_var:>13,.2f} {pct:>13.2f}%")

    # Marginal VaR
    if result.marginal_var:
        print("\n" + "=" * 80)
        print("MARGINAL VaR")
        print("=" * 80)
        for pos_id, marg_var in result.marginal_var.items():
            print(f"{pos_id}: ${marg_var:,.2f}")

    # Incremental VaR
    print("\n" + "=" * 80)
    print("INCREMENTAL VaR ANALYSIS")
    print("=" * 80)

    ivar_result = engine.calculate_incremental_var(portfolio, historical_data)
    print(f"Total Portfolio VaR: ${result.var:,.2f}")
    print(f"Diversification Benefit: ${ivar_result.diversification_benefit:,.2f}")
    print(f"\nPosition-level Incremental VaR:")
    for pos_id, ivari in ivar_result.position_ivari.items():
        var_without = ivar_result.portfolio_var_without_position.get(pos_id, 0)
        print(f"{pos_id}:")
        print(f"  Incremental VaR: ${ivari:,.2f}")
        print(f"  VaR without position: ${var_without:,.2f}")

    # Stressed VaR
    if result.stressed_var:
        print("\n" + "=" * 80)
        print("STRESSED VaR")
        print("=" * 80)
        print(f"Stressed VaR: ${result.stressed_var:,.2f}")
        print(f"Regular VaR: ${result.var:,.2f}")
        print(f"Stress Multiplier: {result.stressed_var / result.var:.2f}x")

    # Performance comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print(f"Calculation Time: {result.execution_time_seconds:.3f} seconds")
    print(f"Positions Processed: {len(portfolio.positions)}")
    print(f"Time per Position: {result.execution_time_seconds / len(portfolio.positions) * 1000:.2f} ms")
    print(f"\nKey Advantage: Parametric VaR is typically 10-100x faster than")
    print(f"historical or Monte Carlo methods, making it ideal for:")
    print(f"  - Real-time risk monitoring")
    print(f"  - Large portfolios (100,000+ positions)")
    print(f"  - Regulatory reporting")
    print(f"  - Stress testing")

    # When to use parametric VaR
    print("\n" + "=" * 80)
    print("WHEN TO USE PARAMETRIC VaR")
    print("=" * 80)
    print("\n✓ BEST FOR:")
    print("  • Large equity portfolios (delta, gamma, vega monitoring)")
    print("  • Fixed income portfolios (DV01, convexity monitoring)")
    print("  • Real-time P&L attribution")
    print("  • Linear products (stocks, forwards, swaps)")
    print("  • Regulatory reporting (sensitivity-based)")
    print("  • portfolios with reliable Greeks")
    print("\n✗ AVOID FOR:")
    print("  • Portfolios with high gamma risk (deep OTM options)")
    print("  • Path-dependent products (barriers, Asians)")
    print("  • Illiquid instruments without reliable Greeks")
    print("  • When capturing fat tails is critical")
    print("  • Complex derivatives without closed-form Greeks")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
```

### 2. Historical VaR Demo (`historical_var_demo.py`)

**Purpose**: Demonstrate full revaluation with historical scenarios - the most accurate method for options and derivatives.

**Key Features to Highlight**:
- Full portfolio revaluation under each scenario
- Captures non-linear effects (gamma, vega, convexity)
- No distributional assumptions
- Worst scenario analysis
- Best for: Options, complex derivatives, portfolios with non-linear payoffs
- Shows actual P&L distribution from historical data

**Detailed Structure**:

```python
"""
Historical VaR (Full Revaluation) Demonstration.

This example demonstrates:
1. Historical VaR calculation using full portfolio revaluation
2. Scenario analysis with worst-case scenarios
3. Capturing non-linear effects (gamma, vega)
4. Comparison with parametric method
5. When to use historical vs other methods

Historical VaR (Full Revaluation Method):
- Revalues portfolio under actual historical market scenarios
- No distributional assumptions about returns
- Captures full non-linear behavior (gamma, vega, convexity)
- Uses actual historical data (spot, vol, rate changes)
- Most accurate method for options and derivatives

Advantages:
+ Most accurate method (uses actual historical data)
+ Handles complex derivatives correctly
+ Captures fat tails and skewness naturally
+ No model risk (e.g., normality assumptions)
+ Full scenario analysis capability

Disadvantages:
- Requires high-quality historical data
- Slower than parametric method
- Limited by historical data length
- May not reflect current market conditions

Mathematical Foundation:
For each historical scenario i:
  P&L_i = Portfolio_Value(Scenario_i) - Portfolio_Value(Current)

VaR = -Percentile(P&L_distribution, α)
where α = 1 - confidence_level

For 99% VaR, VaR is the 1st percentile of the P&L distribution.
"""

def main():
    print("=" * 80)
    print("HISTORICAL VaR (FULL REVALUATION) DEMONSTRATION")
    print("=" * 80)

    # Create portfolio with options
    portfolio = create_sample_portfolio()
    historical_data = generate_historical_data(num_days=500)  # More data for historical

    print(f"\nPortfolio: {portfolio.portfolio_name}")
    print(f"Number of positions: {len(portfolio.positions)}")
    print(f"Portfolio value: ${portfolio.get_portfolio_value():,.2f}")

    # Configure VaR
    config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        lookback_days=252,  # 1 year
        var_method=VaRMethod.HISTORICAL,
        calculate_component_var=True,  # Enable component VaR
        calculate_marginal_var=True,   # Enable marginal VaR
    )

    # Calculate VaR
    print("\n" + "=" * 80)
    print("CALCULATING HISTORICAL VaR")
    print("=" * 80)

    start_time = time.time()

    engine = HistoricalVaREngine(config=config)
    result = engine.calculate_var(portfolio, historical_data)

    calc_time = time.time() - start_time

    print(f"\nCalculation completed in {calc_time:.3f} seconds")
    print(f"Scenarios processed: {len(historical_data)}")
    print(f"Positions repriced per scenario: {len(portfolio.positions)}")

    # Summary report
    summary = VaRReportGenerator.generate_summary(result)
    for key, value in summary.items():
        print(f"{key:30s}: {value}")

    # Worst scenarios
    if hasattr(result, 'worst_scenarios') and result.worst_scenarios:
        print("\n" + "=" * 80)
        print("WORST 10 HISTORICAL SCENARIOS")
        print("=" * 80)
        print(f"{'Rank':<6} {'Scenario Date':<15} {'P&L':>15} {'% of Portfolio':>15}")
        print("-" * 80)
        for i, scenario in enumerate(result.worst_scenarios[:10], 1):
            pnl_pct = (scenario['pnl'] / result.portfolio_value) * 100
            date_str = scenario.get('date', f"Scenario {scenario['scenario_idx']}")
            print(f"{i:<6} {str(date_str):<15} ${scenario['pnl']:>13,.2f} {pnl_pct:>13.2f}%")

    # Scenario analysis
    print("\n" + "=" * 80)
    print("SCENARIO ANALYSIS")
    print("=" * 80)

    # Calculate scenario statistics
    if hasattr(result, 'scenario_pnl'):
        scenario_pnl = result.scenario_pnl
        print(f"Total scenarios: {len(scenario_pnl)}")
        print(f"Mean P&L: ${np.mean(scenario_pnl):,.2f}")
        print(f"Std P&L: ${np.std(scenario_pnl):,.2f}")
        print(f"Min P&L: ${np.min(scenario_pnl):,.2f}")
        print(f"Max P&L: ${np.max(scenario_pnl):,.2f}")
        print(f"Skewness: {scipy.stats.skew(scenario_pnl):.4f}")
        print(f"Kurtosis: {scipy.stats.kurtosis(scenario_pnl):.4f}")

        # Percentiles
        print(f"\nPercentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(scenario_pnl, p)
            print(f"  {p:2d}th percentile: ${val:,.2f}")

    # Component VaR
    if result.component_var:
        print("\n" + "=" * 80)
        print("COMPONENT VaR (POSITION-LEVEL ATTRIBUTION)")
        print("=" * 80)
        print(f"{'Position ID':<20} {'Component VaR':>15} {'% of Total':>15}")
        print("-" * 80)
        sorted_components = sorted(
            result.component_var.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for pos_id, comp_var in sorted_components:
            pct = (comp_var / result.var) * 100
            print(f"{pos_id:<20} ${comp_var:>13,.2f} {pct:>13.2f}%")

    # Incremental VaR
    print("\n" + "=" * 80)
    print("INCREMENTAL VaR ANALYSIS")
    print("=" * 80)

    ivar_result = engine.calculate_incremental_var(portfolio, historical_data)
    print(f"Total Portfolio VaR: ${result.var:,.2f}")
    print(f"Diversification Benefit: ${ivar_result.diversification_benefit:,.2f}")
    print(f"\nDiversification benefit indicates risk reduction from correlation.")
    print(f"Higher benefit = better risk diversification")

    # Comparison with parametric
    print("\n" + "=" * 80)
    print("COMPARISON: HISTORICAL vs PARAMETRIC")
    print("=" * 80)

    # Calculate parametric for comparison
    from var import ParametricVaREngine, EquityRiskFactorConfig

    param_config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        lookback_days=252,
        var_method=VaRMethod.PARAMETRIC,
        equity_factors=EquityRiskFactorConfig(
            include_spot=True,
            include_vol=True,
            include_rate=True,
        ),
    )

    param_engine = ParametricVaREngine(config=param_config)
    param_result = param_engine.calculate_var(portfolio, historical_data)

    print(f"\n{'Metric':<30} {'Historical':>15} {'Parametric':>15} {'Difference':>15}")
    print("-" * 80)
    print(f"{'VaR':<30} ${result.var:>13,.2f} ${param_result.var:>13,.2f} "
          f"${result.var - param_result.var:>13,.2f}")
    print(f"{'CVaR':<30} ${result.cvar:>13,.2f} ${param_result.cvar:>13,.2f} "
          f"${result.cvar - param_result.cvar:>13,.2f}")
    print(f"{'VaR % of Portfolio':<30} {result.var_as_pct * 100:>13.2f}% "
          f"{param_result.var_as_pct * 100:>13.2f}% "
          f"{(result.var_as_pct - param_result.var_as_pct) * 100:>13.2f}%")

    print(f"\nExplanation:")
    print(f"  Historical VaR: ${result.var:,.2f}")
    print(f"  Parametric VaR: ${param_result.var:,.2f}")

    if result.var > param_result.var:
        diff_pct = ((result.var - param_result.var) / param_result.var) * 100
        print(f"\n  Historical VaR is {diff_pct:.1f}% HIGHER than parametric.")
        print(f"  This indicates:")
        print(f"  • Non-linear effects are significant (gamma, vega)")
        print(f"  • Options have meaningful convexity")
        print(f"  • Historical data captures fat tails")
    else:
        diff_pct = ((param_result.var - result.var) / result.var) * 100
        print(f"\n  Parametric VaR is {diff_pct:.1f}% HIGHER than historical.")
        print(f"  This indicates:")
        print(f"  • Portfolio is primarily linear")
        print(f"  • Greeks provide good risk approximation")

    print("\n" + "=" * 80)
    print("WHEN TO USE HISTORICAL VaR")
    print("=" * 80)
    print("\n✓ BEST FOR:")
    print("  • Portfolios with significant gamma risk")
    print("  • Options and complex derivatives")
    print("  • Path-independent options (European, American)")
    print("  • When capturing fat tails is critical")
    print("  • Stress testing with actual historical crises")
    print("  • Portfolios with non-linear payoffs")
    print("  • Model validation and backtesting")

    print("\n✗ AVOID FOR:")
    print("  • Very large portfolios (computational cost)")
    print("  • Path-dependent products (need forward simulation)")
    print("  • When limited historical data available")
    print("  • Real-time risk monitoring (too slow)")
    print("  • Forward-looking stress scenarios")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
```

### 3. Monte Carlo VaR Demo (`monte_carlo_var_demo.py`)

**Purpose**: Demonstrate Monte Carlo simulation with custom distributions and stress scenarios.

**Key Features to Highlight**:
- Simulated scenarios with flexible distributions
- Custom stress scenarios
- Path-dependent products support
- Scenario control and validation
- Best for: Limited data, path-dependent, stress testing
- Most flexible method

**Detailed Structure**:

```python
"""
Monte Carlo VaR (Simulation) Demonstration.

This example demonstrates:
1. Monte Carlo VaR calculation using simulated scenarios
2. Custom distribution modeling (t-distribution, skewed distributions)
3. Stress testing with custom scenarios
4. Path-dependent product simulation
5. Scenario validation and convergence
6. When to use Monte Carlo vs other methods

Monte Carlo VaR (Simulation Method):
- Simulates market scenarios using random draws
- Flexible distribution modeling
- Supports path-dependent products
- Custom stress scenario implementation
- Most flexible VaR method

Advantages:
+ Flexible distribution modeling
+ Supports path-dependent products
+ Forward-looking (can model future scenarios)
+ Stress testing with custom shocks
+ Can incorporate stochastic volatility
+ Handles complex derivatives

Disadvantages:
- Computational intensive
- Convergence issues for tail events
- Requires calibration of distributions
- Random sampling uncertainty
- Model risk (distribution choice)

Mathematical Foundation:
For n simulations:
  Generate random scenarios: S_1, S_2, ..., S_n
  For each scenario i:
    P&L_i = Portfolio_Value(S_i) - Portfolio_Value(Current)
  VaR = -Percentile(P&L_distribution, α)
  where α = 1 - confidence_level

Supports various distributions:
- Normal: N(μ, σ²)
- Student's t: t(ν, μ, σ)
- Skewed t distribution
- GARCH-normal
- Jump-diffusion
"""

import scipy.stats as stats

def generate_custom_scenarios(num_scenarios=10000):
    """Generate Monte Carlo scenarios with different distributions."""
    np.random.seed(42)

    # Normal distribution scenarios
    normal_returns = np.random.normal(0.0005, 0.015, num_scenarios)

    # Student's t-distribution (fat tails)
    t_returns = stats.t.rvs(df=5, loc=0.0005, scale=0.015, size=num_scenarios)

    # Skewed t-distribution
    skewed_returns = stats.skewnorm.rvs(a=2, loc=0.0005, scale=0.015, size=num_scenarios)

    # Create DataFrame
    dates = pd.date_range(start=datetime(2023, 1, 1), periods=num_scenarios, freq="D")

    data = pd.DataFrame(
        {
            "spot_return": normal_returns,
            "vol_change": np.random.normal(0.0, 0.01, num_scenarios),
            "rate_shift": np.random.normal(0.0, 0.0005, num_scenarios),
            "t_spot_return": t_returns,
            "skewed_spot_return": skewed_returns,
        },
        index=dates,
    )

    return data

def main():
    print("=" * 80)
    print("MONTE CARLO VaR (SIMULATION) DEMONSTRATION")
    print("=" * 80)

    # Create portfolio
    portfolio = create_sample_portfolio()
    scenarios = generate_custom_scenarios(num_scenarios=10000)

    print(f"\nPortfolio: {portfolio.portfolio_name}")
    print(f"Number of positions: {len(portfolio.positions)}")
    print(f"Portfolio value: ${portfolio.get_portfolio_value():,.2f}")

    # Configure VaR
    config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        var_method=VaRMethod.MONTE_CARLO,
        mc_num_simulations=10000,
        mc_seed=42,
        calculate_component_var=True,
    )

    # Calculate VaR with different distributions
    print("\n" + "=" * 80)
    print("MONTE CARLO VaR WITH DIFFERENT DISTRIBUTIONS")
    print("=" * 80)

    results = {}
    for dist_name in ["normal", "t_dist", "skewed"]:
        print(f"\nTesting {dist_name} distribution...")

        # Select appropriate column
        if dist_name == "normal":
            data = scenarios[['spot_return', 'vol_change', 'rate_shift']].copy()
        elif dist_name == "t_dist":
            data = scenarios[['t_spot_return', 'vol_change', 'rate_shift']].copy()
            data.rename(columns={'t_spot_return': 'spot_return'}, inplace=True)
        else:  # skewed
            data = scenarios[['skewed_spot_return', 'vol_change', 'rate_shift']].copy()
            data.rename(columns={'skewed_spot_return': 'spot_return'}, inplace=True)

        start_time = time.time()

        engine = MonteCarloVaREngine(config=config)
        result = engine.calculate_var(portfolio, data)

        calc_time = time.time() - start_time
        results[dist_name] = result

        print(f"  VaR: ${result.var:,.2f}")
        print(f"  CVaR: ${result.cvar:,.2f}")
        print(f"  Time: {calc_time:.3f} seconds")

    # Distribution comparison
    print("\n" + "=" * 80)
    print("DISTRIBUTION COMPARISON")
    print("=" * 80)

    print(f"\n{'Distribution':<20} {'VaR':>15} {'CVaR':>15} {'Difference':>15}")
    print("-" * 80)
    base_var = results['normal'].var
    for dist_name, result in results.items():
        diff = result.var - base_var
        print(f"{dist_name.capitalize():<20} ${result.var:>13,.2f} ${result.cvar:>13,.2f} "
              f"${diff:>13,.2f}")

    print(f"\nExplanation:")
    print(f"  • Normal distribution: Standard Black-Scholes assumption")
    print(f"  • t-distribution: Captures fat tails (df=5)")
    print(f"  • Skewed normal: Captures asymmetric risk")

    # Stress testing
    print("\n" + "=" * 80)
    print("STRESS TESTING")
    print("=" * 80)

    # Create stress scenarios
    stress_scenarios = pd.DataFrame({
        'spot_return': [-0.20, -0.15, -0.10, 0.10, 0.15, 0.20],  # ±20% shocks
        'vol_change': [0.10, 0.05, 0.02, -0.02, -0.05, -0.10],    # Volatility jumps
        'rate_shift': [0.005, 0.002, 0.001, -0.001, -0.002, -0.005],  # Rate shifts
    })

    print("\nStress Scenario Portfolio Values:")
    base_value = portfolio.get_portfolio_value()
    print(f"Base Portfolio Value: ${base_value:,.2f}")

    for i, (idx, scenario) in enumerate(stress_scenarios.iterrows()):
        # Apply stress to market data
        stressed_scenarios = scenarios.copy()
        stressed_scenarios['spot_return'] = scenario['spot_return']
        stressed_scenarios['vol_change'] = scenario['vol_change']
        stressed_scenarios['rate_shift'] = scenario['rate_shift']

        # Calculate VaR under stress
        stress_config = VaRConfig(
            confidence_level=0.99,
            holding_period=1,
            var_method=VaRMethod.MONTE_CARLO,
            mc_num_simulations=1000,
        )
        stress_engine = MonteCarloVaREngine(config=stress_config)
        stress_result = stress_engine.calculate_var(portfolio, stressed_scenarios)

        print(f"\nStress {i+1}: Spot={scenario['spot_return']:+.0%}, "
              f"Vol={scenario['vol_change']:+.0%}, Rate={scenario['rate_shift']:+.4f}")
        print(f"  VaR under stress: ${stress_result.var:,.2f}")

    # Convergence analysis
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)

    print("\nMonte Carlo convergence (VaR by number of simulations):")
    n_simulations_list = [1000, 5000, 10000, 20000, 50000]

    for n_sim in n_simulations_list:
        config_mc = VaRConfig(
            confidence_level=0.99,
            holding_period=1,
            var_method=VaRMethod.MONTE_CARLO,
            mc_num_simulations=n_sim,
            mc_seed=42,
        )

        engine_mc = MonteCarloVaREngine(config=config_mc)
        result_mc = engine_mc.calculate_var(portfolio, scenarios)

        print(f"  {n_sim:5d} sims: ${result_mc.var:>10,.2f}")

    # Path-dependent example (Asian option)
    print("\n" + "=" * 80)
    print("PATH-DEPENDENT PRODUCTS")
    print("=" * 80)

    print("\nNote: Monte Carlo is ideal for path-dependent products")
    print("that other methods cannot handle:")
    print("  • Asian options (average price/rate)")
    print("  • Barrier options (knock-in/knock-out)")
    print("  • Lookback options (max/min price)")
    print("  • American options (early exercise)")
    print("  • Basket options (correlated underlyings)")

    print("\nExample path-dependent simulation:")
    # Generate path scenarios
    num_paths = 1000
    num_steps = 252  # Daily steps for 1 year

    print(f"  Generating {num_paths} paths with {num_steps} steps each")
    print(f"  Total price calculations: {num_paths * num_steps:,}")

    # Display distribution statistics
    if hasattr(results['normal'], 'scenario_pnl'):
        pnl = results['normal'].scenario_pnl
        print(f"\nScenario P&L Statistics:")
        print(f"  Mean: ${np.mean(pnl):,.2f}")
        print(f"  Std Dev: ${np.std(pnl):,.2f}")
        print(f"  Skewness: {scipy.stats.skew(pnl):.4f}")
        print(f"  Kurtosis: {scipy.stats.kurtosis(pnl):.4f}")

    print("\n" + "=" * 80)
    print("WHEN TO USE MONTE CARLO VaR")
    print("=" * 80)
    print("\n✓ BEST FOR:")
    print("  • Path-dependent products (Asians, barriers, lookbacks)")
    print("  • When limited historical data available")
    print("  • Custom distribution modeling")
    print("  • Forward-looking stress scenarios")
    print("  • American options with early exercise")
    print("  • Stochastic volatility models")
    print("  • Jump-diffusion processes")
    print("  • Correlation scenarios")

    print("\n✗ AVOID FOR:")
    print("  • Simple linear portfolios (parametric faster)")
    print("  • When many similar calculations needed")
    print("  • Real-time monitoring (too slow)")
    print("  • Very large portfolios without variance reduction")

    print("\nPerformance Tips:")
    print("  • Use variance reduction techniques (antithetic, control variates)")
    print("  • Parallel simulation for large portfolios")
    print("  • Importance sampling for tail events")
    print("  • Quasi-MC (low-discrepancy sequences) for faster convergence")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
```

### 4. VaR Backtest Demo (`var_backtest_demo.py`)

**Purpose**: Demonstrate VaR model validation and backtesting with Kupiec and Christoffersen tests.

**Key Features to Highlight**:
- Kupiec POF (Probability of Failure) test
- Christoffersen independence test
- Basel traffic light regime
- Model accuracy assessment
- Regulatory compliance demonstration
- Visual backtesting results

**Detailed Structure**:

```python
"""
VaR Model Validation and Backtesting Demonstration.

This example demonstrates:
1. Backtesting VaR models using historical P&L data
2. Kupiec POF (Probability of Failure) test
3. Christoffersen independence test
4. Basel traffic light regime
5. Model performance assessment
6. Regulatory compliance (Basel II/III)

Backtesting Framework:
- Tests VaR model accuracy and performance
- Uses actual portfolio P&L data
- Statistical tests for model validation
- Regulatory capital requirements

Kupiec POF Test:
  Tests if observed violations match theoretical expectations

  H0: VaR model is correctly specified
  H1: VaR model is incorrect

  Test statistic: LR_uc ~ χ²(1)
  Critical value at 99%: 6.63

Christoffersen Test:
  Tests independence of violations

  H0: Violations are independent
  H1: Violations show clustering

  Test statistic: LR_ind ~ χ²(1)
  Critical value at 99%: 6.63

Basel Traffic Light:
  Green:    < 5 violations  (99% VaR) - No action
  Yellow:   5-9 violations  - Increased capital charge
  Red:      > 9 violations  - Model review required

Reference: Basel Committee on Banking Supervision (1996, 2013)
"""

def generate_backtest_data(num_days=756):  # 3 years of data
    """Generate realistic backtest data with VaR violations."""
    np.random.seed(42)
    dates = pd.date_range(start=datetime(2021, 1, 1), periods=num_days, freq="D")

    # Generate P&L data with fat tails (some extreme events)
    base_returns = np.random.normal(0.0005, 0.015, num_days)

    # Add crisis periods (higher volatility)
    crisis_periods = [
        (100, 120),   # Crisis 1
        (300, 320),   # Crisis 2
        (500, 520),   # Crisis 3
    ]

    for start, end in crisis_periods:
        crisis_returns = np.random.normal(0, 0.05, end - start)  # Higher volatility
        base_returns[start:end] = crisis_returns

    # Calculate cumulative P&L (start with $1M portfolio)
    portfolio_value = 1_000_000
    pnl = base_returns * portfolio_value

    # Generate VaR estimates (some model backsliding)
    # Assume VaR model with decreasing accuracy
    var_estimates = np.abs(
        np.random.normal(15_000, 2_000, num_days) *  # Base VaR ~1.5%
        (1 + 0.5 * np.random.random(num_days))  # Add some variation
    )

    data = pd.DataFrame({
        'date': dates,
        'pnl': pnl,
        'var_estimate': var_estimates,
        'violation': pnl < -var_estimates,  # True if P&L exceeds VaR
    })

    return data

def perform_kupiec_test(backtest_data, confidence_level=0.99):
    """Perform Kupiec POF test."""
    from scipy.stats import chi2

    num_days = len(backtest_data)
    expected_violations = int((1 - confidence_level) * num_days)
    actual_violations = backtest_data['violation'].sum()

    # Calculate test statistic
    if actual_violations == 0 or actual_violations == num_days:
        lr_uc = np.inf
    else:
        lr_uc = -2 * np.log(
            ((1 - confidence_level) ** (num_days - actual_violations)) *
            (confidence_level ** actual_violations) /
            (((num_days - actual_violations) / num_days) ** (num_days - actual_violations)) *
            ((actual_violations / num_days) ** actual_violations)
        )

    p_value = 1 - chi2.cdf(lr_uc, df=1)

    return {
        'actual_violations': actual_violations,
        'expected_violations': expected_violations,
        'lr_statistic': lr_uc,
        'p_value': p_value,
        'rejection': p_value < (1 - confidence_level),
    }

def perform_christoffersen_test(backtest_data):
    """Perform Christoffersen independence test."""
    from scipy.stats import chi2

    violations = backtest_data['violation'].astype(int).values
    n = len(violations)

    # Count violations and non-violations
    n00 = n10 = n01 = n11 = 0

    for i in range(1, len(violations)):
        if violations[i-1] == 0 and violations[i] == 0:
            n00 += 1
        elif violations[i-1] == 0 and violations[i] == 1:
            n01 += 1
        elif violations[i-1] == 1 and violations[i] == 0:
            n10 += 1
        elif violations[i-1] == 1 and violations[i] == 1:
            n11 += 1

    n0 = n00 + n01  # Previous day no violation
    n1 = n10 + n11  # Previous day violation

    if n0 == 0 or n1 == 0:
        return {
            'lr_statistic': np.inf,
            'p_value': 0.0,
            'rejection': True,
        }

    # Calculate conditional probabilities
    pi01 = n01 / n0 if n0 > 0 else 0  # P(V_t=1 | V_{t-1}=0)
    pi11 = n11 / n1 if n1 > 0 else 0  # P(V_t=1 | V_{t-1}=1)

    # Test statistic
    if pi01 == 0 and pi11 == 0:
        lr_ind = np.inf
    elif pi01 == 0:
        lr_ind = -2 * np.log(
            ((1 - pi11) ** n11) *
            (pi11 ** n10)
        )
    else:
        lr_ind = -2 * np.log(
            ((1 - pi01) ** n00) *
            (pi01 ** n01) *
            ((1 - pi11) ** n11) *
            (pi11 ** n10)
        )

    p_value = 1 - chi2.cdf(lr_ind, df=1)

    return {
        'n00': n00,
        'n01': n01,
        'n10': n10,
        'n11': n11,
        'pi01': pi01,
        'pi11': pi11,
        'lr_statistic': lr_ind,
        'p_value': p_value,
        'rejection': p_value < 0.01,
    }

def get_basel_traffic_light(violations, confidence_level=0.99, total_days=252):
    """Determine Basel traffic light regime."""
    expected_violations = int((1 - confidence_level) * total_days)

    if violations < 5:
        return {
            'regime': 'GREEN',
            'color': 'GREEN',
            'action': 'No action required',
            'description': 'Model performing well'
        }
    elif violations <= 9:
        return {
            'regime': 'YELLOW',
            'color': 'YELLOW',
            'action': 'Increased monitoring and capital charge',
            'action_multiplier': 1.4,
            'description': 'Minor model issues'
        }
    else:
        return {
            'regime': 'RED',
            'color': 'RED',
            'action': 'Model review and potential overhaul',
            'action_multiplier': 3.0,
            'description': 'Significant model deficiencies'
        }

def main():
    print("=" * 80)
    print("VaR MODEL BACKTESTING & VALIDATION DEMONSTRATION")
    print("=" * 80)

    # Generate backtest data
    backtest_data = generate_backtest_data(num_days=756)  # 3 years

    print(f"\nBacktest Period: {backtest_data['date'].min().date()} to {backtest_data['date'].max().date()}")
    print(f"Total Trading Days: {len(backtest_data)}")
    print(f"Portfolio Value: $1,000,000")
    print(f"Confidence Level: 99% (daily VaR)")

    # Summary statistics
    print("\n" + "=" * 80)
    print("PORTFOLIO P&L SUMMARY")
    print("=" * 80)

    total_pnl = backtest_data['pnl'].sum()
    avg_pnl = backtest_data['pnl'].mean()
    std_pnl = backtest_data['pnl'].std()
    max_gain = backtest_data['pnl'].max()
    max_loss = backtest_data['pnl'].min()

    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Average Daily P&L: ${avg_pnl:,.2f}")
    print(f"Daily Volatility: ${std_pnl:,.2f}")
    print(f"Maximum Gain (best day): ${max_gain:,.2f}")
    print(f"Maximum Loss (worst day): ${max_loss:,.2f}")
    print(f"Return/Risk Ratio: {avg_pnl / std_pnl:.4f}")

    # VaR analysis
    print("\n" + "=" * 80)
    print("VaR MODEL SUMMARY")
    print("=" * 80)

    avg_var = backtest_data['var_estimate'].mean()
    violations = backtest_data['violation'].sum()
    violation_rate = violations / len(backtest_data)

    print(f"Average Daily VaR: ${avg_var:,.2f}")
    print(f"VaR as % of Portfolio: {avg_var / 1_000_000:.2%}")
    print(f"Total Violations: {violations}")
    print(f"Violation Rate: {violation_rate:.2%}")
    print(f"Theoretical Rate: 1.00%")

    # Kupiec POF test
    print("\n" + "=" * 80)
    print("KUPREC POF TEST (Unconditional Coverage)")
    print("=" * 80)

    kupiec_result = perform_kupiec_test(backtest_data, confidence_level=0.99)

    print(f"\nNull Hypothesis: VaR model is correctly specified")
    print(f"Alternative Hypothesis: VaR model is incorrect")
    print(f"\nExpected Violations (99%): {kupiec_result['expected_violations']}")
    print(f"Actual Violations: {kupiec_result['actual_violations']}")
    print(f"\nTest Statistic (LR_uc): {kupiec_result['lr_statistic']:.4f}")
    print(f"P-value: {kupiec_result['p_value']:.6f}")
    print(f"Critical Value (99%): 6.63")

    if kupiec_result['rejection']:
        print(f"\n❌ REJECT H0: VaR model is NOT correctly specified")
        print(f"   Model may underestimate risk")
    else:
        print(f"\n✅ FAIL TO REJECT H0: VaR model appears correctly specified")

    # Christoffersen independence test
    print("\n" + "=" * 80)
    print("CHRISTOFFERSEN INDEPENDENCE TEST")
    print("=" * 80)

    christ_result = perform_christoffersen_test(backtest_data)

    print(f"\nNull Hypothesis: Violations are independent")
    print(f"Alternative Hypothesis: Violations show clustering")
    print(f"\nViolation Transition Matrix:")
    print(f"                 Today")
    print(f"                No    Yes")
    print(f"Yesterday No   {christ_result['n00']:3d}   {christ_result['n01']:3d}")
    print(f"       Yes   {christ_result['n10']:3d}   {christ_result['n11']:3d}")

    print(f"\nConditional Probabilities:")
    print(f"P(Violation | Previous No Violation): {christ_result['pi01']:.4f}")
    print(f"P(Violation | Previous Violation): {christ_result['pi11']:.4f}")

    print(f"\nTest Statistic (LR_ind): {christ_result['lr_statistic']:.4f}")
    print(f"P-value: {christ_result['p_value']:.6f}")
    print(f"Critical Value (99%): 6.63")

    if christ_result['rejection']:
        print(f"\n❌ REJECT H0: Violations show clustering")
        print(f"   Risk is time-varying, model not capturing dynamics")
    else:
        print(f"\n✅ FAIL TO REJECT H0: Violations appear independent")

    # Combined test
    print("\n" + "=" * 80)
    print("COMBINED TEST (Kupiec + Christoffersen)")
    print("=" * 80)

    # Calculate joint statistic
    if np.isfinite(kupiec_result['lr_statistic']) and np.isfinite(christ_result['lr_statistic']):
        joint_stat = kupiec_result['lr_statistic'] + christ_result['lr_statistic']
        from scipy.stats import chi2
        joint_p_value = 1 - chi2.cdf(joint_stat, df=2)

        print(f"Combined Test Statistic (LR_cc): {joint_stat:.4f}")
        print(f"P-value: {joint_p_value:.6f}")
        print(f"Critical Value (99%): 9.21")

        if joint_p_value < 0.01:
            print(f"\n❌ COMBINED TEST: REJECT H0")
            print(f"   VaR model has significant deficiencies")
        else:
            print(f"\n✅ COMBINED TEST: FAIL TO REJECT H0")
            print(f"   VaR model passes combined validation")
    else:
        print(f"\nNote: Cannot compute combined test due to infinite components")

    # Basel traffic light
    print("\n" + "=" * 80)
    print("BASEL TRAFFIC LIGHT REGIME")
    print("=" * 80)

    traffic_light = get_basel_traffic_light(violations, confidence_level=0.99, total_days=252)

    print(f"\nViolations in last 252 days: {violations}")
    print(f"Regime: {traffic_light['regime']}")
    print(f"Color: {traffic_light['color']}")
    print(f"\nAction Required: {traffic_light['action']}")

    if traffic_light['regime'] == 'GREEN':
        print(f"\n✅ Model is in compliance with regulatory requirements")
        print(f"   No immediate action needed")
        print(f"   Continue regular monitoring")
    elif traffic_light['regime'] == 'YELLOW':
        print(f"\n⚠️  Model requires attention")
        print(f"   Increase capital charge by 40%")
        print(f"   Enhanced monitoring required")
        print(f"   Consider model improvements")
    else:  # RED
        print(f"\n🚨 Model is non-compliant")
        print(f"   Tripling of capital charge")
        print(f"   Immediate model review required")
        print(f"   Potential model replacement needed")

    # Detailed violation analysis
    print("\n" + "=" * 80)
    print("DETAILED VIOLATION ANALYSIS")
    print("=" * 80)

    violation_dates = backtest_data[backtest_data['violation']]['date']
    print(f"\nViolation Dates ({len(violation_dates)} total):")

    for i, date in enumerate(violation_dates[:10], 1):  # Show first 10
        row = backtest_data[backtest_data['date'] == date].iloc[0]
        print(f"  {i:2d}. {date.date()}: P&L=${row['pnl']:>10,.2f}, VaR=${row['var_estimate']:>10,.2f}")

    if len(violation_dates) > 10:
        print(f"  ... and {len(violation_dates) - 10} more")

    # Check for clustering
    print(f"\nViolation Clustering Analysis:")
    if len(violation_dates) > 1:
        gaps = (violation_dates.diff().dropna().dt.days).astype(int)
        print(f"Average days between violations: {gaps.mean():.1f}")
        print(f"Median days between violations: {gaps.median():.1f}")
        print(f"Min gap: {gaps.min()} days")
        print(f"Max gap: {gaps.max()} days")

        # Consecutive violations
        consecutive = 0
        max_consecutive = 0
        for i in range(1, len(backtest_data)):
            if backtest_data.iloc[i]['violation'] and backtest_data.iloc[i-1]['violation']:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        print(f"Maximum consecutive violations: {max_consecutive}")

    # Model recommendations
    print("\n" + "=" * 80)
    print("MODEL RECOMMENDATIONS")
    print("=" * 80)

    print(f"\nBased on backtesting results:")

    if kupiec_result['rejection']:
        print(f"• Kupiec test failed → Model underestimates risk")
        print(f"  Recommendations:")
        print(f"  - Increase VaR multiplier")
        print(f"  - Use more conservative parameters")
        print(f"  - Add stress testing overlays")
        print(f"  - Consider full revaluation (Historical/MC)")

    if christ_result['rejection']:
        print(f"\n• Christoffersen test failed → Violations cluster")
        print(f"  Recommendations:")
        print(f"  - Implement time-varying volatility (GARCH)")
        print(f"  - Add regime-switching models")
        print(f"  - Increase data frequency")
        print(f"  - Include volatility clustering")

    if not kupiec_result['rejection'] and not christ_result['rejection']:
        print(f"\n✅ Model passes both tests")
        print(f"  Recommendations:")
        print(f"  - Continue regular backtesting")
        print(f"  - Monitor for regime changes")
        print(f"  - Update parameters quarterly")

    # Compare different VaR models
    print("\n" + "=" * 80)
    print("COMPARISON: DIFFERENT VaR MODELS")
    print("=" * 80)

    # Recalculate with parametric and historical for comparison
    portfolio = create_sample_portfolio()
    historical_data = generate_historical_data()

    # Parametric VaR
    param_config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        lookback_days=252,
        var_method=VaRMethod.PARAMETRIC,
    )
    param_engine = ParametricVaREngine(config=param_config)
    param_result = param_engine.calculate_var(portfolio, historical_data)

    # Historical VaR
    hist_config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        lookback_days=252,
        var_method=VaRMethod.HISTORICAL,
    )
    hist_engine = HistoricalVaREngine(config=hist_config)
    hist_result = hist_engine.calculate_var(portfolio, historical_data)

    print(f"\n{'Model':<20} {'VaR':>15} {'CVaR':>15} {'Speed':>15}")
    print("-" * 80)
    print(f"{'Parametric':<20} ${param_result.var:>13,.2f} ${param_result.cvar:>13,.2f} "
          f"{param_result.execution_time_seconds:>13.3f}s")
    print(f"{'Historical':<20} ${hist_result.var:>13,.2f} ${hist_result.cvar:>13,.2f} "
          f"{hist_result.execution_time_seconds:>13.3f}s")
    print(f"{'Backtest Model':<20} ${avg_var:>13,.2f} N/A "
          f"{'N/A':>15}")

    print(f"\nRecommendation for this portfolio:")
    print(f"  • If portfolio is linear: Use Parametric (fastest)")
    print(f"  • If portfolio has options: Use Historical (most accurate)")
    print(f"  • For regulatory reporting: Validate with backtesting")

    print("\n" + "=" * 80)
    print("BACKTESTING COMPLETE")
    print("=" * 80)

    print(f"\nRegulatory Summary:")
    print(f"  • Expected violations: {kupiec_result['expected_violations']}")
    print(f"  • Actual violations: {kupiec_result['actual_violations']}")
    print(f"  • Basel regime: {traffic_light['regime']}")
    print(f"  • Model status: {'PASS' if not kupiec_result['rejection'] else 'FAIL'}")

    return 0
```

## Testing and Validation Approach

### Unit Testing Strategy

Each example script should be testable with pytest. Create corresponding test files:

```python
# test/test_parametric_var_demo.py
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from example.parametric_var_demo import (
    create_sample_portfolio,
    generate_historical_data,
    main
)

class TestParametricVaRDemo:
    """Test parametric VaR demonstration script."""

    def test_create_sample_portfolio(self):
        """Test portfolio creation."""
        portfolio = create_sample_portfolio()
        assert portfolio is not None
        assert len(portfolio.positions) > 0
        assert portfolio.portfolio_name is not None

    def test_generate_historical_data(self):
        """Test historical data generation."""
        data = generate_historical_data(num_days=100)
        assert len(data) == 100
        assert 'spot_return' in data.columns
        assert 'vol_change' in data.columns
        assert 'rate_shift' in data.columns

    @patch('builtins.print')
    def test_main_execution(self, mock_print):
        """Test main execution completes without errors."""
        # This should run without throwing exceptions
        result = main()
        assert result == 0

    @patch('var.engines.parametric.ParametricVaREngine.calculate_var')
    def test_var_calculation(self, mock_calc):
        """Test VaR calculation is called with correct parameters."""
        from var.results import VaRResult

        # Mock return value
        mock_result = VaRResult(
            var=15000.0,
            cvar=20000.0,
            confidence_level=0.99,
            holding_period=1,
            method=VaRMethod.PARAMETRIC,
            portfolio_value=100000.0,
            var_as_pct=0.15,
            calculation_timestamp=datetime.now(),
            execution_time_seconds=0.1,
        )
        mock_calc.return_value = mock_result

        portfolio = create_sample_portfolio()
        data = generate_historical_data()

        # VaR calculation should be triggered
        # (actual test would check the call)
```

### Integration Testing

Test the complete workflow:

1. **Script execution test** - Run each script and verify it completes without errors
2. **VaR calculation test** - Verify VaR values are calculated and are reasonable
3. **Report generation test** - Verify reports are generated with expected content
4. **Data validation test** - Verify historical data is valid and sufficient

### Validation Criteria

For each script:

1. **Correctness**:
   - VaR values are positive and reasonable (e.g., 1-5% of portfolio value)
   - CVaR >= VaR
   - Component VaR sum equals total VaR (within tolerance)
   - Factor attribution percentages sum to 100%

2. **Performance**:
   - Parametric VaR: < 1 second for portfolio with 10 positions
   - Historical VaR: < 5 seconds for same portfolio
   - Monte Carlo VaR: < 10 seconds for 10,000 simulations
   - Backtesting: < 3 seconds for 756 days

3. **Educational Value**:
   - Clear explanation of method
   - When-to-use guidance
   - Comparison with other methods
   - Practical insights

### Example Test File Template

```python
# test/test_var_demos.py
"""Test VaR example demonstrations."""

import pytest
from datetime import datetime
import numpy as np

from example.parametric_var_demo import (
    create_sample_portfolio as create_parametric_portfolio,
    main as parametric_main
)
from example.historical_var_demo import (
    create_sample_portfolio as create_historical_portfolio,
    main as historical_main
)
from example.monte_carlo_var_demo import (
    create_sample_portfolio as create_mc_portfolio,
    main as mc_main
)
from example.var_backtest_demo import (
    generate_backtest_data,
    main as backtest_main
)

class TestParametricVaRDemo:
    """Test parametric VaR demo."""

    def test_portfolio_creation(self):
        portfolio = create_parametric_portfolio()
        assert len(portfolio.positions) == 2
        assert portfolio.get_portfolio_value() > 0

    def test_var_calculation(self):
        portfolio = create_parametric_portfolio()
        data = generate_historical_data(300)
        assert len(data) == 300

    def test_main_runs(self):
        # Test that main executes without errors
        result = parametric_main()
        assert result == 0

class TestHistoricalVaRDemo:
    """Test historical VaR demo."""

    def test_portfolio_creation(self):
        portfolio = create_historical_portfolio()
        assert len(portfolio.positions) == 2

    def test_main_runs(self):
        result = historical_main()
        assert result == 0

class TestMonteCarloVaRDemo:
    """Test Monte Carlo VaR demo."""

    def test_portfolio_creation(self):
        portfolio = create_mc_portfolio()
        assert len(portfolio.positions) == 2

    def test_main_runs(self):
        result = mc_main()
        assert result == 0

class TestVaRBacktestDemo:
    """Test VaR backtest demo."""

    def test_backtest_data_generation(self):
        data = generate_backtest_data(100)
        assert len(data) == 100
        assert 'pnl' in data.columns
        assert 'var_estimate' in data.columns
        assert 'violation' in data.columns

    def test_main_runs(self):
        result = backtest_main()
        assert result == 0

# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Educational Value

Each example script will teach users:

### 1. Parametric VaR Demo
**Learning Objectives**:
- Understand variance-covariance method
- Learn about sensitivity-based risk calculation
- Understand component, factor, and marginal VaR
- Know when to use parametric for speed vs accuracy trade-offs
- Appreciate attribution analysis for risk decomposition

**Key Takeaways**:
- Fastest method for large portfolios
- Requires reliable Greeks
- Best for linear products
- Supports comprehensive attribution
- Industry standard for trading desks

### 2. Historical VaR Demo
**Learning Objectives**:
- Understand full revaluation method
- Learn about scenario-based risk calculation
- See non-linear effects in action
- Understand worst-case scenario analysis
- Know when historical data is available

**Key Takeaways**:
- Most accurate method for options
- No distributional assumptions
- Captures fat tails naturally
- Requires quality historical data
- Best for non-linear products

### 3. Monte Carlo VaR Demo
**Learning Objectives**:
- Understand simulation-based methods
- Learn about custom distribution modeling
- Understand path-dependent products
- Learn stress testing techniques
- Know convergence analysis

**Key Takeaways**:
- Most flexible method
- Supports path-dependency
- Custom distributions possible
- Computationally intensive
- Best for complex products

### 4. Backtest Demo
**Learning Objectives**:
- Understand VaR model validation
- Learn Kupiec and Christoffersen tests
- Understand Basel regulations
- Learn traffic light regime
- Know model improvement strategies

**Key Takeaways**:
- Models must be validated
- Statistical tests for verification
- Regulatory compliance requirements
- Model degradation over time
- Continuous monitoring needed

## Implementation Checklist

### Before Implementation
- [ ] Review existing codebase patterns
- [ ] Understand each VaR engine's capabilities
- [ ] Plan code structure and organization
- [ ] Prepare example data scenarios

### During Implementation
- [ ] Create `parametric_var_demo.py`
  - [ ] Portfolio setup with options
  - [ ] VaR configuration with attribution flags
  - [ ] Component VaR display
  - [ ] Factor VaR display
  - [ ] Marginal VaR display
  - [ ] Incremental VaR analysis
  - [ ] Stressed VaR demonstration
  - [ ] Performance metrics
  - [ ] When-to-use guidance

- [ ] Create `historical_var_demo.py`
  - [ ] Portfolio setup with options
  - [ ] Historical VaR calculation
  - [ ] Worst scenarios analysis
  - [ ] Scenario statistics (skewness, kurtosis)
  - [ ] Component VaR display
  - [ ] Incremental VaR analysis
  - [ ] Comparison with parametric
  - [ ] When-to-use guidance

- [ ] Create `monte_carlo_var_demo.py`
  - [ ] Portfolio setup
  - [ ] Multiple distribution scenarios (normal, t, skewed)
  - [ ] Stress testing scenarios
  - [ ] Convergence analysis
  - [ ] Path-dependent product discussion
  - [ ] Performance tips
  - [ ] When-to-use guidance

- [ ] Create `var_backtest_demo.py`
  - [ ] Backtest data generation
  - [ ] Kupiec POF test implementation
  - [ ] Christoffersen independence test
  - [ ] Basel traffic light regime
  - [ ] Detailed violation analysis
  - [ ] Model recommendations
  - [ ] Regulatory summary

- [ ] Create test files for each script
  - [ ] Unit tests for helper functions
  - [ ] Integration tests for main execution
  - [ ] Validation tests for VaR calculations

### After Implementation
- [ ] Run all scripts manually to verify output
- [ ] Execute test suite
- [ ] Check educational content clarity
- [ ] Verify documentation
- [ ] Update example index (README)
- [ ] Submit for review

## Dependencies

All scripts will use the following dependencies (already in requirements.txt):

```
numpy
pandas
scipy
pytest
```

Additional imports from QuantArk:
- `asset.equity.*`
- `var.*`
- `portfolio.*`
- `priceenv.*`
- `param.*`
- `util.*`

## Performance Benchmarks

Expected performance for reference portfolio (2 options positions):

| Method | Expected Time | Relative Speed |
|--------|---------------|----------------|
| Parametric | < 0.5s | 1x (baseline) |
| Historical | < 3s | ~6x slower |
| Monte Carlo | < 8s | ~16x slower |
| Backtesting | < 3s | ~6x slower |

## Error Handling

Each script should handle:
- Empty portfolio
- Insufficient historical data
- Invalid configuration parameters
- Numerical convergence issues

Error messages should be informative and guide users to resolution.

## Documentation Standards

Each script must have:
- Module-level docstring explaining purpose and usage
- Function-level docstrings for all public functions
- Inline comments explaining complex calculations
- Clear section headers in output
- Educational explanations in output

## Conclusion

This implementation plan provides a comprehensive roadmap for creating 4 focused VaR example scripts that will educate users on:
- Each VaR method's strengths and weaknesses
- When to use each method
- How to interpret results
- Best practices for risk management
- Regulatory compliance requirements

Each script follows established patterns, provides educational value, and demonstrates practical usage of the QuantArk VaR module.
