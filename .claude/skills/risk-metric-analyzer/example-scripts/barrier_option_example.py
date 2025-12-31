"""
Risk Metric Analysis Example: Barrier Option
=============================================

This script demonstrates risk metric analysis for a barrier option
using QuantArk's PDE solver. Barrier options require PDE methods
for accurate Greeks calculation.

Usage:
    python risk_metric_analysis/risk_metric_calculation_scripts/barrier_option_example.py
"""
from datetime import datetime
import pandas as pd
import os
import numpy as np

# QuantArk imports
from asset.equity.product.option import BarrierOption
from asset.equity.engine.pde import BarrierPDESolver
from asset.equity.engine import PDEEngine
from asset.equity.riskmeasures import GreeksCalculator
from asset.equity.param import EngineParams, PDEParams
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType, BarrierType

# === Configuration ===
# Product Parameters
STRIKE = 100.0
BARRIER = 90.0     # Down barrier
MATURITY = 1.0     # 1 year
OPTION_TYPE = OptionType.CALL
BARRIER_TYPE = BarrierType.DOWN_AND_OUT  # Knock-out if spot hits 90

# Market Data
SPOT = 100.0       # 10% above barrier
VOLATILITY = 0.25  # 25% implied vol
RATE = 0.05        # 5% risk-free rate
DIV_YIELD = 0.02   # 2% dividend yield

# Output directory
OUTPUT_DIR = 'risk_metric_analysis/reports/'


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

    # Create barrier option
    option = BarrierOption(
        strike=STRIKE,
        barrier=BARRIER,
        option_type=OPTION_TYPE,
        barrier_type=BARRIER_TYPE,
        maturity=MATURITY,
        rebate=0.0,  # No rebate on knock-out
    )

    # Create PDE engine (uses BarrierPDESolver internally)
    engine = PDEEngine()

    return option, pricing_env, engine


def calculate_greeks(option, pricing_env, engine):
    """Calculate numerical Greeks using PDE."""

    calculator = GreeksCalculator(params=EngineParams(bump_size=0.01))

    # Price
    price = engine.price(option, pricing_env)

    # Numerical Greeks (no analytical available for barriers)
    numerical_greeks = calculator.calculate_numerical_greeks(option, pricing_env, engine)

    return price, numerical_greeks


def analyze_barrier_proximity(option, pricing_env, engine):
    """Analyze Greeks sensitivity to barrier proximity."""

    calculator = GreeksCalculator()
    results = []

    # Distance to barrier (spot / barrier)
    for proximity in [1.02, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]:
        from copy import deepcopy
        from param import SpotQuote
        env = deepcopy(pricing_env)
        env.spot_quote = SpotQuote(spot=BARRIER * proximity)  # Recreate to trigger validation

        try:
            greeks = calculator.calculate_numerical_greeks(option, env, engine)
            results.append({
                'Spot/Barrier': proximity,
                'Spot': env.spot_quote.spot,
                'Distance to Barrier': env.spot_quote.spot - BARRIER,
                'Price': greeks['price'],
                'Delta': greeks['delta'],
                'Gamma': greeks['gamma'],
                'Vega': greeks['vega'],
            })
        except Exception as e:
            print(f"Warning: Could not calculate at proximity {proximity}: {e}")

    return pd.DataFrame(results)


def print_report(price, greeks, proximity_df):
    """Print formatted risk report."""

    print("=" * 60)
    print("RISK METRIC ANALYSIS REPORT")
    print("Barrier Option (Down-and-Out Call)")
    print("=" * 60)
    print()

    print("PRODUCT SPECIFICATION")
    print("-" * 40)
    print(f"  Strike:         {STRIKE:,.2f}")
    print(f"  Barrier:        {BARRIER:,.2f}")
    print(f"  Barrier Type:   {BARRIER_TYPE.name}")
    print(f"  Maturity:       {MATURITY:.2f} years")
    print(f"  Option Type:    {OPTION_TYPE.name}")
    print()

    print("MARKET DATA")
    print("-" * 40)
    print(f"  Spot:           {SPOT:,.2f}")
    print(f"  Spot/Barrier:   {SPOT/BARRIER:.2%}")
    print(f"  Volatility:     {VOLATILITY:.2%}")
    print(f"  Rate:           {RATE:.2%}")
    print(f"  Div Yield:      {DIV_YIELD:.2%}")
    print()

    print("PRICING RESULTS")
    print("-" * 40)
    print(f"  Barrier Option Price: {price:,.6f}")

    # Compare to vanilla (no barrier)
    from asset.equity.product.option import EuropeanVanillaOption
    from asset.equity.engine.analytical import BlackScholesEngine

    vanilla = EuropeanVanillaOption(
        strike=STRIKE,
        option_type=OPTION_TYPE,
        maturity=MATURITY
    )
    vanilla_engine = BlackScholesEngine()
    from copy import deepcopy
    env = deepcopy(pricing_env)
    vanilla_price = vanilla_engine.price(vanilla, env)

    print(f"  Vanilla Price:        {vanilla_price:,.6f}")
    print(f"  Barrier Discount:     {(1 - price/vanilla_price):,.2%}")
    print()

    print("GREEKS (NUMERICAL PDE)")
    print("-" * 40)
    print(f"  Delta (Δ):    {greeks['delta']:,.6f}")
    print(f"  Gamma (Γ):    {greeks['gamma']:,.6f}")
    print(f"  Vega (ν):     {greeks['vega']:,.6f} (per 1% vol)")
    print(f"  Theta (Θ):    {greeks['theta']:,.6f} (per day)")
    print(f"  Rho (ρ):      {greeks['rho']:,.6f} (per 1% rate)")
    print()

    print("BARRIER PROXIMITY ANALYSIS")
    print("-" * 40)
    if not proximity_df.empty:
        print(proximity_df.to_string(index=False))
    print()

    print("KEY INSIGHTS")
    print("-" * 40)
    print(f"  1. As spot approaches barrier ({BARRIER}), option value decreases rapidly")
    print(f"  2. Delta is lower than vanilla due to knock-out risk")
    print(f"  3. Gamma can become very high near the barrier")
    print(f"  4. Barrier discount reflects knockout probability: {(1 - price/vanilla_price):,.2%}")
    print()


def save_results(greeks, proximity_df):
    """Save results to CSV."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save Greeks
    base_df = pd.DataFrame([{
        'Product': 'Barrier Call (Down-Out)',
        'Strike': STRIKE,
        'Barrier': BARRIER,
        'BarrierType': BARRIER_TYPE.name,
        'Maturity': MATURITY,
        'Spot': SPOT,
        'Volatility': VOLATILITY,
        **greeks,
        'Timestamp': datetime.now().isoformat(),
    }])

    base_path = os.path.join(OUTPUT_DIR, 'barrier_option_greeks.csv')
    base_df.to_csv(base_path, index=False)
    print(f"Saved: {base_path}")

    # Save proximity analysis
    if not proximity_df.empty:
        proximity_path = os.path.join(OUTPUT_DIR, 'barrier_option_proximity.csv')
        proximity_df.to_csv(proximity_path, index=False)
        print(f"Saved: {proximity_path}")


def main():
    """Main execution."""

    print("\n" + "=" * 60)
    print("Running Barrier Option Risk Analysis...")
    print("=" * 60 + "\n")

    # Setup
    option, pricing_env, engine = setup()

    # Calculate Greeks
    price, greeks = calculate_greeks(option, pricing_env, engine)

    # Analyze barrier proximity
    proximity_df = analyze_barrier_proximity(option, pricing_env, engine)

    # Print report
    print_report(price, greeks, proximity_df)

    # Save results
    save_results(greeks, proximity_df)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
