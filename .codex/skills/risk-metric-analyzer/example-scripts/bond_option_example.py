"""
Risk Metric Analysis Example: Bond Option
==========================================

This script demonstrates risk metric analysis for a European bond option
using QuantArk's Black '76 engine. Includes bond-specific sensitivities
like DV01 and duration.

Usage:
    python risk_metric_analysis/risk_metric_calculation_scripts/bond_option_example.py
"""
from datetime import datetime, date
import pandas as pd
import os

# QuantArk imports
from asset.bond.product.couponbond import FixedBond
from asset.bond.product.option import EuroShortTermBondOption
from asset.bond.engine.analytical import BlackBondOptionEngine
from asset.bond.engine.discount import BondDiscountEngine
from asset.bond.riskmeasures import BondGreeksCalculator
from param import SpotQuote, FlatVolSurface, FlatRateCurve
from priceenv import PricingEnvironment
from util.enum import OptionType

# === Configuration ===
# Bond Parameters
COUPON_RATE = 0.05      # 5% annual coupon
FACE_VALUE = 100.0
COUPON_FREQUENCY = 2    # Semi-annual
BOND_MATURITY = date(2029, 6, 15)  # 5 years from now

# Option Parameters
STRIKE = 102.0          # Slight OTM call
OPTION_EXPIRY = date(2025, 6, 15)  # 1 year option
OPTION_TYPE = OptionType.CALL
NOTIONAL = 1_000_000.0  # $1M notional

# Market Data
RATE = 0.04             # 4% flat rate curve
VOLATILITY = 0.08       # 8% price volatility
VALUATION_DATE = date(2024, 6, 15)

# Output directory
OUTPUT_DIR = 'risk_metric_analysis/reports/'


def setup():
    """Create underlying bond, option, pricing environment, and engines."""

    # Create pricing environment
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),  # Placeholder
        vol_surface=FlatVolSurface(volatility=VOLATILITY),
        rate_curve=FlatRateCurve(rate=RATE),
        valuation_date=datetime.combine(VALUATION_DATE, datetime.min.time()),
    )

    # Create underlying bond
    underlying = FixedBond(
        face_value=FACE_VALUE,
        coupon_rate=COUPON_RATE,
        coupon_frequency=COUPON_FREQUENCY,
        issue_date=date(2024, 6, 15),
        maturity_date=BOND_MATURITY,
    )

    # Create bond option
    option = EuroShortTermBondOption(
        underlying=underlying,
        strike=STRIKE,
        expiry_date=OPTION_EXPIRY,
        option_type=OPTION_TYPE,
        notional=NOTIONAL,
    )

    # Create engines
    option_engine = BlackBondOptionEngine(pricing_env)
    bond_engine = BondDiscountEngine(pricing_env)

    return underlying, option, pricing_env, option_engine, bond_engine


def calculate_greeks(option, pricing_env, volatility=None):
    """Calculate analytical and numerical Greeks for bond option."""

    calculator = BondGreeksCalculator(bump_size=0.01)

    # Analytical Greeks (Black '76)
    analytical_greeks = calculator.calculate_analytical_greeks(
        option, pricing_env, volatility=volatility
    )

    # Numerical Greeks (FDM)
    numerical_greeks = calculator.calculate_numerical_greeks(
        option, pricing_env, volatility=volatility
    )

    # Bond-specific sensitivities
    bond_sensitivities = calculator.calculate_bond_sensitivities(
        option, pricing_env, volatility=volatility
    )

    # Compare analytical vs numerical
    comparison = calculator.compare_greeks(analytical_greeks, numerical_greeks)

    return analytical_greeks, numerical_greeks, bond_sensitivities, comparison


def calculate_underlying_metrics(underlying, pricing_env, bond_engine):
    """Calculate underlying bond metrics."""

    valuation_date = pricing_env.valuation_date

    metrics = {}

    # Price
    dirty_price = bond_engine.dirty_price(underlying, valuation_date, valuation_date)
    clean_price = bond_engine.clean_price(underlying, valuation_date, valuation_date)
    accrued = bond_engine.accrued_interest(underlying, valuation_date)

    metrics['dirty_price'] = dirty_price
    metrics['clean_price'] = clean_price
    metrics['accrued_interest'] = accrued

    # Duration and DV01
    metrics['modified_duration'] = bond_engine.modified_duration(
        underlying, valuation_date, valuation_date
    )
    metrics['dv01'] = bond_engine.dv01(underlying, valuation_date, valuation_date)

    # YTM
    metrics['ytm'] = bond_engine.yield_to_maturity(
        underlying, dirty_price, valuation_date
    )

    return metrics


def print_report(underlying_metrics, analytical_greeks, numerical_greeks,
                 bond_sensitivities, comparison):
    """Print formatted risk report."""

    print("=" * 70)
    print("RISK METRIC ANALYSIS REPORT")
    print("European Bond Option (Black '76)")
    print("=" * 70)
    print()

    print("UNDERLYING BOND")
    print("-" * 50)
    print(f"  Coupon Rate:      {COUPON_RATE:.2%}")
    print(f"  Face Value:       {FACE_VALUE:,.2f}")
    print(f"  Coupon Frequency: {COUPON_FREQUENCY}x per year")
    print(f"  Maturity Date:    {BOND_MATURITY}")
    print(f"  Clean Price:      {underlying_metrics['clean_price']:,.4f}")
    print(f"  Dirty Price:      {underlying_metrics['dirty_price']:,.4f}")
    print(f"  Accrued Interest: {underlying_metrics['accrued_interest']:,.4f}")
    print(f"  YTM:              {underlying_metrics['ytm']:.4%}")
    print(f"  Modified Duration:{underlying_metrics['modified_duration']:.4f}")
    print(f"  DV01:             {underlying_metrics['dv01']:,.4f}")
    print()

    print("OPTION SPECIFICATION")
    print("-" * 50)
    print(f"  Strike:           {STRIKE:,.2f}")
    print(f"  Expiry Date:      {OPTION_EXPIRY}")
    print(f"  Option Type:      {OPTION_TYPE.name}")
    print(f"  Notional:         {NOTIONAL:,.0f}")
    print(f"  Price Volatility: {VOLATILITY:.2%}")
    print()

    print("OPTION PRICING RESULTS")
    print("-" * 50)
    print(f"  Option Price:     {analytical_greeks['price']:,.2f}")
    print(f"  Forward Price:    {analytical_greeks['forward_price']:,.4f}")
    print(f"  Moneyness:        {analytical_greeks['moneyness']:.4f}")
    print()

    print("OPTION GREEKS (ANALYTICAL - Black '76)")
    print("-" * 50)
    print(f"  Delta (Δ):        {analytical_greeks['delta']:,.2f}")
    print(f"  Gamma (Γ):        {analytical_greeks['gamma']:,.2f}")
    print(f"  Vega (ν):         {analytical_greeks['vega']:,.2f} (per 1% vol)")
    print(f"  Theta (Θ):        {analytical_greeks['theta']:,.2f} (per day)")
    print(f"  Rho (ρ):          {analytical_greeks['rho']:,.2f} (per 1% rate)")
    print()

    print("BOND-SPECIFIC SENSITIVITIES")
    print("-" * 50)
    print(f"  Option Price:       {bond_sensitivities['option_price']:,.2f}")
    print(f"  Underlying Price:   {bond_sensitivities['underlying_price']:,.4f}")
    print(f"  Option DV01:        {bond_sensitivities['option_dv01']:,.4f}")
    print(f"  Option Duration:    {bond_sensitivities['option_duration']:.4f}")
    print(f"  Underlying DV01:    {bond_sensitivities['underlying_dv01']:,.4f}")
    print(f"  Underlying Duration:{bond_sensitivities['underlying_duration']:.4f}")
    print(f"  Delta-Equiv DV01:   {bond_sensitivities['delta_equivalent_dv01']:,.4f}")
    print()

    print("GREEKS COMPARISON (Analytical vs Numerical)")
    print("-" * 50)
    for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        if greek in comparison['difference']:
            diff = comparison['difference'][greek]
            print(f"  {greek.capitalize():8s}: Abs Diff = {diff['absolute']:+.4f}, Rel Diff = {diff['relative']:+.2%}")
    print()

    print("INTERPRETATION")
    print("-" * 50)
    delta = analytical_greeks['delta']
    delta_per_unit = delta / NOTIONAL

    print(f"  1. Delta: Option gains {delta:,.0f} for 1 point bond price increase")
    print(f"     Per-unit delta: {delta_per_unit:.4f}")
    print(f"  2. DV01: Option loses {bond_sensitivities['option_dv01']:,.2f} for 1bp rate increase")
    print(f"  3. Duration-equivalent exposure: {bond_sensitivities['delta_equivalent_dv01']:,.2f}")
    print()


def save_results(underlying_metrics, analytical_greeks, bond_sensitivities):
    """Save results to CSV."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Combine all metrics
    results = {
        'Product': 'Bond Option Call',
        'Strike': STRIKE,
        'OptionExpiry': str(OPTION_EXPIRY),
        'BondMaturity': str(BOND_MATURITY),
        'Notional': NOTIONAL,
        'Volatility': VOLATILITY,
        'Rate': RATE,
        # Underlying
        'Bond_CleanPrice': underlying_metrics['clean_price'],
        'Bond_DirtyPrice': underlying_metrics['dirty_price'],
        'Bond_YTM': underlying_metrics['ytm'],
        'Bond_ModDuration': underlying_metrics['modified_duration'],
        'Bond_DV01': underlying_metrics['dv01'],
        # Option Greeks
        'Option_Price': analytical_greeks['price'],
        'Option_Delta': analytical_greeks['delta'],
        'Option_Gamma': analytical_greeks['gamma'],
        'Option_Vega': analytical_greeks['vega'],
        'Option_Theta': analytical_greeks['theta'],
        'Option_Rho': analytical_greeks['rho'],
        'Forward_Price': analytical_greeks['forward_price'],
        # Bond sensitivities
        'Option_DV01': bond_sensitivities['option_dv01'],
        'Option_Duration': bond_sensitivities['option_duration'],
        'Delta_Equiv_DV01': bond_sensitivities['delta_equivalent_dv01'],
        'Timestamp': datetime.now().isoformat(),
    }

    df = pd.DataFrame([results])
    output_path = os.path.join(OUTPUT_DIR, 'bond_option_greeks.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    """Main execution."""

    print("\n" + "=" * 70)
    print("Running Bond Option Risk Analysis...")
    print("=" * 70 + "\n")

    # Setup
    underlying, option, pricing_env, option_engine, bond_engine = setup()

    # Calculate underlying metrics
    underlying_metrics = calculate_underlying_metrics(underlying, pricing_env, bond_engine)

    # Calculate option Greeks
    analytical, numerical, bond_sens, comparison = calculate_greeks(
        option, pricing_env, volatility=VOLATILITY
    )

    # Print report
    print_report(underlying_metrics, analytical, numerical, bond_sens, comparison)

    # Save results
    save_results(underlying_metrics, analytical, bond_sens)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
