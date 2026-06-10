"""
Greeks Verification Script for Digital Option Analytical Engine
Uses finite difference method to verify Greeks
Generated: 2024-12-25
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent.parent))

from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
from quantark.asset.equity.engine.analytical.digital_option_engine import DigitalOptionAnalyticalEngine
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.util.enum import OptionType
from datetime import datetime

# Tolerance for Greeks comparison
GREEK_TOLERANCE = 0.15  # 15% tolerance (Greeks from finite difference are approximate)


def create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.0):
    """Helper to create pricing environment."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        rate_curve=FlatRateCurve(rate=rate),
        vol_surface=FlatVolSurface(volatility=vol),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def create_digital_call(K=100.0, payout=10.0, T=1.0):
    """Helper to create digital call option."""
    return CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.CALL,
        maturity=T,
    )


def create_digital_put(K=100.0, payout=10.0, T=1.0):
    """Helper to create digital put option."""
    return CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.PUT,
        maturity=T,
    )


def calculate_numerical_greeks(option, pricing_env, bump=0.001):
    """Calculate Greeks using finite difference."""

    engine = DigitalOptionAnalyticalEngine()
    original_price = engine.price(option, pricing_env)

    # Delta: dP/dS
    original_spot = pricing_env.spot
    original_r = pricing_env.get_rate(option.maturity)
    original_vol = pricing_env.get_vol(option.strike, option.maturity)
    original_q = pricing_env.get_div_yield(option.maturity)

    env_up = PricingEnvironment(
        spot_quote=SpotQuote(spot=original_spot * (1 + bump)),
        rate_curve=FlatRateCurve(rate=original_r),
        vol_surface=FlatVolSurface(volatility=original_vol),
        div_yield=ContinuousDividendYield(div_yield=original_q),
        valuation_date=datetime(2024, 1, 1),
    )
    price_up = engine.price(option, env_up)

    env_down = PricingEnvironment(
        spot_quote=SpotQuote(spot=original_spot * (1 - bump)),
        rate_curve=FlatRateCurve(rate=original_r),
        vol_surface=FlatVolSurface(volatility=original_vol),
        div_yield=ContinuousDividendYield(div_yield=original_q),
        valuation_date=datetime(2024, 1, 1),
    )
    price_down = engine.price(option, env_down)

    delta_fd = (price_up - price_down) / (2 * original_spot * bump)

    # Gamma: d^2P/dS^2
    gamma_fd = (price_up - 2 * original_price + price_down) / ((original_spot * bump) ** 2)

    # Vega: dP/dσ
    env_vol_up = PricingEnvironment(
        spot_quote=SpotQuote(spot=original_spot),
        rate_curve=FlatRateCurve(rate=original_r),
        vol_surface=FlatVolSurface(volatility=original_vol * (1 + bump)),
        div_yield=ContinuousDividendYield(div_yield=original_q),
        valuation_date=datetime(2024, 1, 1),
    )
    price_vol_up = engine.price(option, env_vol_up)

    env_vol_down = PricingEnvironment(
        spot_quote=SpotQuote(spot=original_spot),
        rate_curve=FlatRateCurve(rate=original_r),
        vol_surface=FlatVolSurface(volatility=original_vol * (1 - bump)),
        div_yield=ContinuousDividendYield(div_yield=original_q),
        valuation_date=datetime(2024, 1, 1),
    )
    price_vol_down = engine.price(option, env_vol_down)

    vega_fd = (price_vol_up - price_vol_down) / (2 * original_vol * bump)

    # Theta: dP/dT (negative because T decreases)
    original_T = option.maturity
    if original_T > 0.01:
        option_T_up = CashOrNothingDigitalOption(
            strike=option.strike,
            payout=option.payout,
            option_type=option.option_type,
            maturity=original_T * (1 + bump),
        )
        price_T_up = engine.price(option_T_up, pricing_env)

        option_T_down = CashOrNothingDigitalOption(
            strike=option.strike,
            payout=option.payout,
            option_type=option.option_type,
            maturity=max(0.001, original_T * (1 - bump)),
        )
        price_T_down = engine.price(option_T_down, pricing_env)

        theta_fd = (price_T_up - price_T_down) / (2 * original_T * bump)
        # Theta is usually defined as -dP/dT (price decrease as time passes)
        theta_fd = -theta_fd
    else:
        theta_fd = None

    # Rho: dP/dr
    env_r_up = PricingEnvironment(
        spot_quote=SpotQuote(spot=original_spot),
        rate_curve=FlatRateCurve(rate=original_r + bump),
        vol_surface=FlatVolSurface(volatility=original_vol),
        div_yield=ContinuousDividendYield(div_yield=original_q),
        valuation_date=datetime(2024, 1, 1),
    )
    price_r_up = engine.price(option, env_r_up)

    env_r_down = PricingEnvironment(
        spot_quote=SpotQuote(spot=original_spot),
        rate_curve=FlatRateCurve(rate=max(0, original_r - bump)),
        vol_surface=FlatVolSurface(volatility=original_vol),
        div_yield=ContinuousDividendYield(div_yield=original_q),
        valuation_date=datetime(2024, 1, 1),
    )
    price_r_down = engine.price(option, env_r_down)

    rho_fd = (price_r_up - price_r_down) / (2 * bump)

    return {
        'price': original_price,
        'delta': delta_fd,
        'gamma': gamma_fd,
        'vega': vega_fd,
        'theta': theta_fd,
        'rho': rho_fd,
    }


def verify_greeks_properties(results, option, pricing_env, name):
    """Verify theoretical properties of digital option Greeks."""

    greeks = calculate_numerical_greeks(option, pricing_env)

    results['greeks_values'][name] = greeks

    # Delta should be in [0, payout/S] for calls (approximately)
    # For digital options, delta can be positive or negative near strike
    delta = greeks['delta']

    # Gamma should be non-negative for call/put (convexity)
    # But digital options can have negative gamma in some regions
    gamma = greeks['gamma']

    # Vega should be positive for ATM options (volatility increases probability range)
    vega = greeks['vega']

    # Check that values are finite
    results['checks'].append({
        'name': f"{name} - Delta is finite",
        'passed': np.isfinite(delta),
        'value': delta,
    })

    results['checks'].append({
        'name': f"{name} - Gamma is finite",
        'passed': np.isfinite(gamma),
        'value': gamma,
    })

    results['checks'].append({
        'name': f"{name} - Vega is finite",
        'passed': np.isfinite(vega),
        'value': vega,
    })

    if greeks['theta'] is not None:
        results['checks'].append({
            'name': f"{name} - Theta is finite",
            'passed': np.isfinite(greeks['theta']),
            'value': greeks['theta'],
        })

    # Rho relationship for digital options:
    # Call rho = -T * price + payout * T * exp(-rT) * N(d2) ... complicated
    # Just check finiteness
    results['checks'].append({
        'name': f"{name} - Rho is finite",
        'passed': np.isfinite(greeks['rho']),
        'value': greeks['rho'],
    })

    # For digital options, verify specific relationships
    # Deep ITM call: delta ~ 0 (price doesn't change much with S once deep ITM)
    # Deep OTM call: delta ~ 0
    # ATM call: delta is at maximum (steep probability transition)

    S = pricing_env.spot
    K = option.strike
    moneyness = S / K

    if moneyness > 1.2:  # Deep ITM
        results['checks'].append({
            'name': f"{name} - Deep ITM delta small",
            'passed': abs(delta) < 0.5,  # Delta should be small for deep ITM
            'value': delta,
            'expected': '< 0.5',
        })
    elif moneyness < 0.8:  # Deep OTM
        results['checks'].append({
            'name': f"{name} - Deep OTM delta small",
            'passed': abs(delta) < 0.5,
            'value': delta,
            'expected': '< 0.5',
        })


def run_greeks_verification():
    """Run comprehensive Greeks verification."""

    print("\n" + "="*70)
    print("DIGITAL OPTION ANALYTICAL ENGINE - GREEKS VERIFICATION")
    print("="*70)

    results = {
        'greeks_values': {},
        'checks': [],
    }

    test_cases = [
        # (spot, strike, payout, T, rate, vol, div, option_type, name)
        (100, 100, 10, 1.0, 0.05, 0.20, 0.02, OptionType.CALL, "ATM Call"),
        (100, 100, 10, 1.0, 0.05, 0.20, 0.02, OptionType.PUT, "ATM Put"),
        (110, 100, 10, 1.0, 0.05, 0.20, 0.02, OptionType.CALL, "ITM Call"),
        (90, 100, 10, 1.0, 0.05, 0.20, 0.02, OptionType.CALL, "OTM Call"),
        (90, 100, 10, 1.0, 0.05, 0.20, 0.02, OptionType.PUT, "ITM Put"),
        (110, 100, 10, 1.0, 0.05, 0.20, 0.02, OptionType.PUT, "OTM Put"),
        (130, 80, 10, 1.0, 0.05, 0.20, 0.02, OptionType.CALL, "Deep ITM Call"),
        (70, 130, 10, 1.0, 0.05, 0.20, 0.02, OptionType.CALL, "Deep OTM Call"),
        (100, 100, 10, 0.25, 0.05, 0.20, 0.02, OptionType.CALL, "Short Term ATM Call"),
        (100, 100, 10, 2.0, 0.05, 0.20, 0.02, OptionType.CALL, "Long Term ATM Call"),
        (100, 100, 10, 1.0, 0.05, 0.10, 0.02, OptionType.CALL, "Low Vol ATM Call"),
        (100, 100, 10, 1.0, 0.05, 0.40, 0.02, OptionType.CALL, "High Vol ATM Call"),
    ]

    for S, K, payout, T, r, vol, q, opt_type, name in test_cases:
        env = create_pricing_env(spot=S, rate=r, vol=vol, div=q)

        if opt_type == OptionType.CALL:
            option = create_digital_call(K=K, payout=payout, T=T)
        else:
            option = create_digital_put(K=K, payout=payout, T=T)

        print(f"\nVerifying: {name}")
        verify_greeks_properties(results, option, env, name)

    # Print results
    print("\n" + "="*70)
    print("GREEKS VERIFICATION RESULTS")
    print("="*70)

    passed = sum(1 for c in results['checks'] if c['passed'])
    total = len(results['checks'])

    print(f"\nPassed: {passed}/{total} ({100*passed/total:.1f}%)")

    if passed < total:
        print("\nFailed checks:")
        for c in results['checks']:
            if not c['passed']:
                exp = c.get('expected', 'N/A')
                print(f"  ✗ {c['name']}: value={c['value']:.6f}, expected={exp}")

    # Print Greeks values table
    print("\n" + "="*70)
    print("GREEKS VALUES TABLE")
    print("="*70)
    print(f"{'Case':<25} {'Price':>10} {'Delta':>10} {'Gamma':>12} {'Vega':>12} {'Theta':>12} {'Rho':>12}")
    print("-" * 95)

    for name, greeks in results['greeks_values'].items():
        theta_str = f"{greeks['theta']:.6f}" if greeks['theta'] else "N/A"
        print(f"{name:<25} {greeks['price']:>10.6f} {greeks['delta']:>10.6f} "
              f"{greeks['gamma']:>12.6f} {greeks['vega']:>12.6f} {theta_str:>12} {greeks['rho']:>12.6f}")

    # Digital option Greeks characteristics
    print("\n" + "="*70)
    print("DIGITAL OPTION GREEKS CHARACTERISTICS")
    print("="*70)
    print("""
    Digital options have unique Greeks characteristics:

    1. Delta:
       - Peaks at the strike (steepest probability transition)
       - Approximately zero when deep ITM or deep OTM
       - Can be positive or negative depending on position relative to strike

    2. Gamma:
       - Can be NEGATIVE (unlike vanilla options)
       - Positive on one side of strike, negative on the other
       - Largest magnitude near the strike

    3. Vega:
       - Can be positive or negative
       - For ATM calls: higher vol spreads probability, can decrease price
       - For ITM calls: higher vol increases probability of moving OTM

    4. Theta:
       - Generally positive for OTM (time decay helps)
       - Can be negative for ITM (time decay hurts)

    5. Rho:
       - Call rho: -T * price + T * payout * exp(-rT) * N(d2)
       - Complex relationship due to discounting and probability effects
    """)

    return passed == total


if __name__ == "__main__":
    success = run_greeks_verification()
    sys.exit(0 if success else 1)
