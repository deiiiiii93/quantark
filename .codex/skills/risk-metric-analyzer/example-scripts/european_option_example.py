"""
Risk Metric Analysis Example: European Vanilla Option
=====================================================

This script demonstrates comprehensive risk metric analysis for a European
vanilla option using QuantArk's analytical Black-Scholes engine.

Usage:
    python risk_metric_analysis/risk_metric_calculation_scripts/european_option_example.py
"""
from datetime import datetime
import pandas as pd
import os

# QuantArk imports
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.riskmeasures import GreeksCalculator
from asset.equity.param import EngineParams
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType

# === Configuration ===
# Product Parameters
STRIKE = 100.0
MATURITY = 1.0  # 1 year
OPTION_TYPE = OptionType.CALL

# Market Data
SPOT = 100.0       # ATM option
VOLATILITY = 0.20  # 20% implied vol
RATE = 0.05        # 5% risk-free rate
DIV_YIELD = 0.02   # 2% dividend yield

# Output directory
OUTPUT_DIR = 'risk_metric_analysis/reports/'

# === Setup ===
def setup():
    """Create product, pricing environment, and engine."""

    # Create pricing environment
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=SPOT),
        vol_surface=FlatVolSurface(volatility=VOLATILITY),
        rate_curve=FlatRateCurve(rate=RATE),
        div_yield=ContinuousDividendYield(div_yield=DIV_YIELD),
        valuation_date=datetime.now(),
    )

    # Create product
    option = EuropeanVanillaOption(
        strike=STRIKE,
        option_type=OPTION_TYPE,
        maturity=MATURITY
    )

    # Create engine
    engine = BlackScholesEngine()

    return option, pricing_env, engine


def calculate_greeks(option, pricing_env, engine):
    """Calculate analytical and numerical Greeks."""

    calculator = GreeksCalculator(params=EngineParams(bump_size=0.01))

    # Analytical Greeks (exact Black-Scholes)
    analytical_greeks = calculator.calculate_analytical_greeks(option, pricing_env)

    # Numerical Greeks (finite difference)
    numerical_greeks = calculator.calculate_numerical_greeks(option, pricing_env, engine)

    # Compare
    comparison = calculator.compare_greeks(analytical_greeks, numerical_greeks)

    return analytical_greeks, numerical_greeks, comparison


def generate_scenario_analysis(option, pricing_env, engine):
    """Generate scenario analysis for spot and volatility changes."""

    calculator = GreeksCalculator()
    scenarios = []

    # Spot scenarios
    spot_changes = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]

    for pct in spot_changes:
        from copy import deepcopy
        from param import SpotQuote
        env = deepcopy(pricing_env)
        env.spot_quote = SpotQuote(spot=SPOT * (1 + pct))  # Recreate to trigger validation

        greeks = calculator.calculate_analytical_greeks(option, env)

        scenarios.append({
            'Scenario': f'Spot {pct:+.0%}',
            'Spot': env.spot_quote.spot,
            'Volatility': VOLATILITY,
            'Price': greeks['price'],
            'Delta': greeks['delta'],
            'Gamma': greeks['gamma'],
            'Vega': greeks['vega'],
            'Theta': greeks['theta'],
            'Rho': greeks['rho'],
        })

    # Volatility scenarios
    vol_changes = [-0.05, -0.02, 0.0, 0.02, 0.05]

    for delta_vol in vol_changes:
        from copy import deepcopy
        env = deepcopy(pricing_env)
        env.vol_surface = FlatVolSurface(volatility=VOLATILITY + delta_vol)

        greeks = calculator.calculate_analytical_greeks(option, env)

        scenarios.append({
            'Scenario': f'Vol {delta_vol:+.0%}',
            'Spot': SPOT,
            'Volatility': VOLATILITY + delta_vol,
            'Price': greeks['price'],
            'Delta': greeks['delta'],
            'Gamma': greeks['gamma'],
            'Vega': greeks['vega'],
            'Theta': greeks['theta'],
            'Rho': greeks['rho'],
        })

    return pd.DataFrame(scenarios)


def print_report(analytical_greeks, numerical_greeks, comparison):
    """Print formatted risk report."""

    print("=" * 60)
    print("RISK METRIC ANALYSIS REPORT")
    print("European Vanilla Call Option")
    print("=" * 60)
    print()

    print("PRODUCT SPECIFICATION")
    print("-" * 40)
    print(f"  Strike:       {STRIKE:,.2f}")
    print(f"  Maturity:     {MATURITY:.2f} years")
    print(f"  Option Type:  {OPTION_TYPE.name}")
    print()

    print("MARKET DATA")
    print("-" * 40)
    print(f"  Spot:         {SPOT:,.2f}")
    print(f"  Volatility:   {VOLATILITY:.2%}")
    print(f"  Rate:         {RATE:.2%}")
    print(f"  Div Yield:    {DIV_YIELD:.2%}")
    print()

    print("PRICING RESULTS")
    print("-" * 40)
    print(f"  Price:        {analytical_greeks['price']:,.6f}")
    intrinsic = max(0, SPOT - STRIKE)
    time_value = analytical_greeks['price'] - intrinsic
    print(f"  Intrinsic:    {intrinsic:,.6f}")
    print(f"  Time Value:   {time_value:,.6f}")
    print()

    print("GREEKS (ANALYTICAL)")
    print("-" * 40)
    print(f"  Delta (Δ):    {analytical_greeks['delta']:,.6f}")
    print(f"  Gamma (Γ):    {analytical_greeks['gamma']:,.6f}")
    print(f"  Vega (ν):     {analytical_greeks['vega']:,.6f} (per 1% vol)")
    print(f"  Theta (Θ):    {analytical_greeks['theta']:,.6f} (per day)")
    print(f"  Rho (ρ):      {analytical_greeks['rho']:,.6f} (per 1% rate)")
    print()

    print("GREEKS COMPARISON (Analytical vs Numerical)")
    print("-" * 40)
    for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        diff = comparison['difference'][greek]
        print(f"  {greek.capitalize():8s}: Abs Diff = {diff['absolute']:+.2e}, Rel Diff = {diff['relative']:+.2%}")
    print()

    print("INTERPRETATION")
    print("-" * 40)
    delta = analytical_greeks['delta']
    print(f"  Delta: Position changes ${delta * 100:,.2f} for every $1 spot move")
    print(f"  Hedge: Sell {delta * 100:,.0f} shares to delta-hedge")

    gamma = analytical_greeks['gamma']
    print(f"  Gamma: Delta changes by {gamma:,.6f} for every $1 spot move")

    vega = analytical_greeks['vega']
    print(f"  Vega: Price changes ${vega:,.4f} for 1% vol increase")

    theta = analytical_greeks['theta']
    print(f"  Theta: Loses ${abs(theta):,.4f} per day (${abs(theta * 7):,.4f}/week)")
    print()


def save_results(analytical_greeks, scenarios_df):
    """Save results to CSV files."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save base case Greeks
    base_df = pd.DataFrame([{
        'Product': 'European Call',
        'Strike': STRIKE,
        'Maturity': MATURITY,
        'OptionType': OPTION_TYPE.name,
        'Spot': SPOT,
        'Volatility': VOLATILITY,
        'Rate': RATE,
        'DivYield': DIV_YIELD,
        **analytical_greeks,
        'Timestamp': datetime.now().isoformat(),
    }])

    base_path = os.path.join(OUTPUT_DIR, 'european_call_greeks.csv')
    base_df.to_csv(base_path, index=False)
    print(f"Saved: {base_path}")

    # Save scenario analysis
    scenarios_path = os.path.join(OUTPUT_DIR, 'european_call_scenarios.csv')
    scenarios_df.to_csv(scenarios_path, index=False)
    print(f"Saved: {scenarios_path}")


def main():
    """Main execution."""

    print("\n" + "=" * 60)
    print("Running Risk Metric Analysis...")
    print("=" * 60 + "\n")

    # Setup
    option, pricing_env, engine = setup()

    # Calculate Greeks
    analytical, numerical, comparison = calculate_greeks(option, pricing_env, engine)

    # Generate scenarios
    scenarios_df = generate_scenario_analysis(option, pricing_env, engine)

    # Print report
    print_report(analytical, numerical, comparison)

    # Save results
    save_results(analytical, scenarios_df)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
