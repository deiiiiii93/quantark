"""
Demonstration of European bond option pricing with QuantArk.

This example shows:
1. Creating a European bond option on a fixed-rate bond
2. Pricing the option using the Black '76 model
3. Calculating Greeks (delta, gamma, vega, theta, rho)
4. Sensitivity analysis (volatility, rates)
5. Put-call parity verification
6. Bond-specific risk measures (DV01, duration)
"""

from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import QuantArk modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.bond.product.couponbond.fixed_bond import FixedBond, create_simple_fixed_bond
from asset.bond.product.option.euro_short_term_bond_option import (
    EuroShortTermBondOption,
    create_bond_option,
)
from asset.bond.engine.analytical.black_engine import BlackBondOptionEngine
from asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from asset.bond.riskmeasures.bond_greeks_calculator import BondGreeksCalculator
from param.rrf.rate_curve import FlatRateCurve
from param.vol import FlatVolSurface
from priceenv import PricingEnvironment
from util.enum import OptionType, PaymentFrequency
from util.calendar import DayCountConvention


def example_1_basic_option_pricing():
    """Example 1: Basic bond option pricing."""
    print("=" * 80)
    print("Example 1: Basic European Bond Option Pricing")
    print("=" * 80)
    
    # Create underlying bond (5-year Treasury-like bond)
    underlying = create_simple_fixed_bond(
        issue_date=datetime(2023, 1, 1),
        maturity_date=datetime(2028, 1, 1),
        notional=1000.0,
        coupon_rate=0.05,  # 5% annual coupon
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        day_count_convention=DayCountConvention.ACT_ACT_ISDA
    )
    
    print(f"Underlying Bond: {underlying}")
    print(f"  Coupon: {underlying.coupon_rate:.2%} semi-annual")
    print(f"  Maturity: {underlying.maturity_date.date()}")
    
    # Create call option expiring in 1 year
    call_option = EuroShortTermBondOption(
        underlying=underlying,
        strike=1000.0,  # At-the-money strike
        expiry_date=datetime(2025, 1, 1),
        option_type=OptionType.CALL,
        notional=1.0,
        strike_is_clean=True
    )
    
    print(f"\nCall Option: {call_option}")
    print(f"  Strike: ${call_option.strike:.2f}")
    print(f"  Expiry: {call_option.expiry_date.date()}")
    
    # Set up pricing environment
    valuation_date = datetime(2024, 1, 1)
    rate_curve = FlatRateCurve(rate=0.04)  # 4% flat curve
    vol_surface = FlatVolSurface(volatility=0.10)  # 10% price volatility
    
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date,
        vol_surface=vol_surface
    )
    
    # Price the underlying bond first
    bond_engine = BondDiscountEngine(pricing_env)
    bond_clean = bond_engine.clean_price(underlying, valuation_date, valuation_date)
    bond_dirty = bond_engine.dirty_price(underlying, valuation_date, valuation_date)
    
    print(f"\nUnderlying Bond Price as of {valuation_date.date()}:")
    print(f"  Clean Price: ${bond_clean:.2f}")
    print(f"  Dirty Price: ${bond_dirty:.2f}")
    
    # Price the option
    engine = BlackBondOptionEngine(pricing_env)
    results = engine.price_with_details(call_option, volatility=0.10)
    
    print(f"\nBlack '76 Model Results:")
    print(f"  Option Price: ${results.price:.4f}")
    print(f"  Forward Bond Price: ${results.forward_bond_price:.2f}")
    print(f"  Time to Expiry: {results.time_to_expiry:.4f} years")
    print(f"  Discount Factor: {results.discount_factor:.6f}")
    print(f"  d1: {results.d1:.4f}")
    print(f"  d2: {results.d2:.4f}")
    
    print()


def example_2_greeks_calculation():
    """Example 2: Calculate and analyze option Greeks."""
    print("=" * 80)
    print("Example 2: Bond Option Greeks")
    print("=" * 80)
    
    # Create underlying and option
    underlying = create_simple_fixed_bond(
        issue_date=datetime(2023, 1, 1),
        maturity_date=datetime(2028, 1, 1),
        notional=1000.0,
        coupon_rate=0.05,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL
    )
    
    option = EuroShortTermBondOption(
        underlying=underlying,
        strike=1000.0,
        expiry_date=datetime(2025, 1, 1),
        option_type=OptionType.CALL
    )
    
    valuation_date = datetime(2024, 1, 1)
    rate_curve = FlatRateCurve(rate=0.04)
    vol_surface = FlatVolSurface(volatility=0.10)
    
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date,
        vol_surface=vol_surface
    )
    
    calculator = BondGreeksCalculator()
    
    # Calculate analytical Greeks
    analytical = calculator.calculate_analytical_greeks(
        option, pricing_env, volatility=0.10
    )
    
    print("Analytical Greeks (Black '76 Model):")
    print(f"  Price:       ${analytical['price']:.4f}")
    print(f"  Delta:       {analytical['delta']:.6f}")
    print(f"  Gamma:       {analytical['gamma']:.8f}")
    print(f"  Vega:        ${analytical['vega']:.4f} (per 1% vol)")
    print(f"  Theta:       ${analytical['theta']:.4f} (per day)")
    print(f"  Rho:         ${analytical['rho']:.4f} (per 1% rate)")
    print(f"  Forward:     ${analytical['forward_price']:.2f}")
    print(f"  Moneyness:   {analytical['moneyness']:.4f}")
    
    # Calculate numerical Greeks for comparison
    numerical = calculator.calculate_numerical_greeks(
        option, pricing_env, volatility=0.10
    )
    
    print("\nNumerical Greeks (Finite Difference):")
    print(f"  Price:       ${numerical['price']:.4f}")
    print(f"  Delta:       {numerical['delta']:.6f}")
    print(f"  Gamma:       {numerical['gamma']:.8f}")
    print(f"  Vega:        ${numerical['vega']:.4f} (per 1% vol)")
    print(f"  Theta:       ${numerical['theta']:.4f} (per day)")
    print(f"  Rho:         ${numerical['rho']:.4f} (per 1% rate)")
    print(f"  DV01:        ${numerical['dv01']:.4f} (per 1bp)")
    
    # Compare methods
    comparison = calculator.compare_greeks(analytical, numerical)
    
    print("\nDifference (Analytical - Numerical):")
    for key in ["price", "delta", "vega"]:
        if key in comparison["difference"]:
            diff = comparison["difference"][key]
            print(f"  {key}: abs={diff['absolute']:.6f}, rel={diff['relative']:.4%}")
    
    print()


def example_3_put_call_parity():
    """Example 3: Verify put-call parity."""
    print("=" * 80)
    print("Example 3: Put-Call Parity Verification")
    print("=" * 80)
    
    underlying = create_simple_fixed_bond(
        issue_date=datetime(2023, 1, 1),
        maturity_date=datetime(2028, 1, 1),
        notional=1000.0,
        coupon_rate=0.05,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL
    )
    
    strike = 1000.0
    expiry_date = datetime(2025, 1, 1)
    volatility = 0.10
    
    call = EuroShortTermBondOption(
        underlying=underlying,
        strike=strike,
        expiry_date=expiry_date,
        option_type=OptionType.CALL
    )
    
    put = EuroShortTermBondOption(
        underlying=underlying,
        strike=strike,
        expiry_date=expiry_date,
        option_type=OptionType.PUT
    )
    
    valuation_date = datetime(2024, 1, 1)
    rate_curve = FlatRateCurve(rate=0.04)
    
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    
    engine = BlackBondOptionEngine(pricing_env)
    
    call_price = engine.price(call, volatility=volatility)
    put_price = engine.price(put, volatility=volatility)
    
    # Get forward and discount factor
    results = engine.price_with_details(call, volatility=volatility)
    F = results.forward_bond_price
    D = results.discount_factor
    
    print(f"Strike: ${strike:.2f}")
    print(f"Call Price: ${call_price:.4f}")
    print(f"Put Price: ${put_price:.4f}")
    print(f"Forward Price: ${F:.2f}")
    print(f"Discount Factor: {D:.6f}")
    
    # Put-Call Parity: C - P = D * (F - K)
    lhs = call_price - put_price
    rhs = D * (F - strike)
    
    print(f"\nPut-Call Parity Check:")
    print(f"  C - P = ${lhs:.4f}")
    print(f"  D * (F - K) = ${rhs:.4f}")
    print(f"  Difference: ${abs(lhs - rhs):.6f}")
    print(f"  Parity holds: {'Yes' if abs(lhs - rhs) < 0.01 else 'No'}")
    
    print()


def example_4_volatility_sensitivity():
    """Example 4: Analyze sensitivity to volatility."""
    print("=" * 80)
    print("Example 4: Volatility Sensitivity Analysis")
    print("=" * 80)
    
    underlying = create_simple_fixed_bond(
        issue_date=datetime(2023, 1, 1),
        maturity_date=datetime(2028, 1, 1),
        notional=1000.0,
        coupon_rate=0.05,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL
    )
    
    option = EuroShortTermBondOption(
        underlying=underlying,
        strike=1000.0,
        expiry_date=datetime(2025, 1, 1),
        option_type=OptionType.CALL
    )
    
    valuation_date = datetime(2024, 1, 1)
    rate_curve = FlatRateCurve(rate=0.04)
    
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    
    engine = BlackBondOptionEngine(pricing_env)
    
    print("Volatility | Call Price | Delta    | Vega")
    print("-" * 50)
    
    calculator = BondGreeksCalculator()
    
    for vol_pct in [5, 7.5, 10, 12.5, 15, 20]:
        vol = vol_pct / 100
        greeks = calculator.calculate_analytical_greeks(
            option, pricing_env, volatility=vol
        )
        
        print(f"  {vol_pct:5.1f}%    |  ${greeks['price']:7.2f} | {greeks['delta']:8.4f} | ${greeks['vega']:6.2f}")
    
    print()


def example_5_rate_sensitivity():
    """Example 5: Analyze sensitivity to interest rates."""
    print("=" * 80)
    print("Example 5: Interest Rate Sensitivity Analysis")
    print("=" * 80)
    
    underlying = create_simple_fixed_bond(
        issue_date=datetime(2023, 1, 1),
        maturity_date=datetime(2028, 1, 1),
        notional=1000.0,
        coupon_rate=0.05,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL
    )
    
    option = EuroShortTermBondOption(
        underlying=underlying,
        strike=1000.0,
        expiry_date=datetime(2025, 1, 1),
        option_type=OptionType.CALL
    )
    
    valuation_date = datetime(2024, 1, 1)
    volatility = 0.10
    
    print("Rate   | Bond Price | Forward  | Call Price | Rho")
    print("-" * 60)
    
    calculator = BondGreeksCalculator()
    
    for rate_pct in [2, 3, 4, 5, 6]:
        rate = rate_pct / 100
        rate_curve = FlatRateCurve(rate=rate)
        
        pricing_env = PricingEnvironment(
            rate_curve=rate_curve,
            valuation_date=valuation_date
        )
        
        bond_engine = BondDiscountEngine(pricing_env)
        bond_price = bond_engine.clean_price(underlying, valuation_date, valuation_date)
        
        engine = BlackBondOptionEngine(pricing_env)
        results = engine.price_with_details(option, volatility=volatility)
        
        greeks = calculator.calculate_analytical_greeks(
            option, pricing_env, volatility=volatility
        )
        
        print(f" {rate_pct:4.1f}%  | ${bond_price:9.2f} | ${results.forward_bond_price:7.2f} | ${results.price:9.2f} | ${greeks['rho']:6.2f}")
    
    print()


def example_6_bond_risk_measures():
    """Example 6: Calculate bond-specific risk measures."""
    print("=" * 80)
    print("Example 6: Bond-Specific Risk Measures")
    print("=" * 80)
    
    underlying = create_simple_fixed_bond(
        issue_date=datetime(2023, 1, 1),
        maturity_date=datetime(2028, 1, 1),
        notional=1000.0,
        coupon_rate=0.05,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL
    )
    
    option = EuroShortTermBondOption(
        underlying=underlying,
        strike=1000.0,
        expiry_date=datetime(2025, 1, 1),
        option_type=OptionType.CALL,
        notional=100  # 100 contracts
    )
    
    valuation_date = datetime(2024, 1, 1)
    rate_curve = FlatRateCurve(rate=0.04)
    
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    
    calculator = BondGreeksCalculator()
    
    sensitivities = calculator.calculate_bond_sensitivities(
        option, pricing_env, volatility=0.10
    )
    
    print("Underlying Bond:")
    print(f"  Price: ${sensitivities['underlying_price']:.2f}")
    print(f"  DV01: ${sensitivities['underlying_dv01']:.4f}")
    print(f"  Modified Duration: {sensitivities['underlying_duration']:.4f} years")
    
    print("\nOption Position (100 contracts):")
    print(f"  Price: ${sensitivities['option_price']:.2f}")
    print(f"  DV01: ${sensitivities['option_dv01']:.4f}")
    print(f"  Effective Duration: {sensitivities['option_duration']:.4f}")
    print(f"  Delta-Equivalent DV01: ${sensitivities['delta_equivalent_dv01']:.4f}")
    
    print()


def example_7_strike_analysis():
    """Example 7: Analyze options across different strikes."""
    print("=" * 80)
    print("Example 7: Strike Analysis")
    print("=" * 80)
    
    underlying = create_simple_fixed_bond(
        issue_date=datetime(2023, 1, 1),
        maturity_date=datetime(2028, 1, 1),
        notional=1000.0,
        coupon_rate=0.05,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL
    )
    
    valuation_date = datetime(2024, 1, 1)
    rate_curve = FlatRateCurve(rate=0.04)
    
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    
    engine = BlackBondOptionEngine(pricing_env)
    bond_engine = BondDiscountEngine(pricing_env)
    
    # Get current bond price for reference
    bond_price = bond_engine.clean_price(underlying, valuation_date, valuation_date)
    
    print(f"Current Bond Price: ${bond_price:.2f}")
    print()
    print("Strike  | Moneyness | Call     | Put      | Intrinsic (Call)")
    print("-" * 65)
    
    volatility = 0.10
    
    for strike in [900, 950, 1000, 1050, 1100]:
        call = EuroShortTermBondOption(
            underlying=underlying,
            strike=float(strike),
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        put = EuroShortTermBondOption(
            underlying=underlying,
            strike=float(strike),
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.PUT
        )
        
        call_results = engine.price_with_details(call, volatility=volatility)
        put_price = engine.price(put, volatility=volatility)
        
        moneyness = call_results.forward_bond_price / strike
        intrinsic = max(call_results.forward_bond_price - strike, 0)
        
        status = "ITM" if moneyness > 1 else ("ATM" if abs(moneyness - 1) < 0.02 else "OTM")
        
        print(f"${strike:6.0f} | {moneyness:9.4f} | ${call_results.price:7.2f} | ${put_price:7.2f} | ${intrinsic:7.2f} ({status})")
    
    print()


def example_8_implied_volatility():
    """Example 8: Calculate implied volatility from market prices."""
    print("=" * 80)
    print("Example 8: Implied Volatility Calculation")
    print("=" * 80)
    
    underlying = create_simple_fixed_bond(
        issue_date=datetime(2023, 1, 1),
        maturity_date=datetime(2028, 1, 1),
        notional=1000.0,
        coupon_rate=0.05,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL
    )
    
    valuation_date = datetime(2024, 1, 1)
    rate_curve = FlatRateCurve(rate=0.04)
    
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    
    engine = BlackBondOptionEngine(pricing_env)
    
    # Simulate market prices at known volatilities
    print("Recovering implied volatility from market prices:\n")
    print("True Vol | Market Price | Implied Vol | Error")
    print("-" * 50)
    
    for true_vol in [0.05, 0.08, 0.10, 0.12, 0.15]:
        option = EuroShortTermBondOption(
            underlying=underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        # Get "market" price at true vol
        market_price = engine.price(option, volatility=true_vol)
        
        # Recover implied vol
        implied_vol = engine.implied_volatility(
            option,
            market_price,
            initial_guess=0.10
        )
        
        error = abs(implied_vol - true_vol) * 10000  # in basis points
        
        print(f" {true_vol:6.2%}  |   ${market_price:8.4f} |   {implied_vol:6.2%}  | {error:5.1f} bps")
    
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("EUROPEAN BOND OPTION PRICING DEMONSTRATION")
    print("Using Black '76 Model")
    print("=" * 80 + "\n")
    
    example_1_basic_option_pricing()
    example_2_greeks_calculation()
    example_3_put_call_parity()
    example_4_volatility_sensitivity()
    example_5_rate_sensitivity()
    example_6_bond_risk_measures()
    example_7_strike_analysis()
    example_8_implied_volatility()
    
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

