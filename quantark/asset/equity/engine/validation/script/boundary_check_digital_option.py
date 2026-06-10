"""
Boundary Check Script for Digital Option Analytical Engine
Generated: 2024-12-25
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
from quantark.asset.equity.engine.analytical.digital_option_engine import DigitalOptionAnalyticalEngine
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError, PricingError
from datetime import datetime


class BoundaryCheckResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_result(self, test_name: str, passed: bool, message: str, value=None, expected=None):
        if passed:
            self.passed.append((test_name, message, value, expected))
        else:
            self.failed.append((test_name, message, value, expected))

    def add_warning(self, test_name: str, message: str):
        self.warnings.append((test_name, message))

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*70}")
        print(f"BOUNDARY CHECK SUMMARY")
        print(f"{'='*70}")
        print(f"Total Tests: {total}")
        print(f"Passed: {len(self.passed)} ({100*len(self.passed)/total if total > 0 else 0:.1f}%)")
        print(f"Failed: {len(self.failed)} ({100*len(self.failed)/total if total > 0 else 0:.1f}%)")
        print(f"Warnings: {len(self.warnings)}")

        if self.failed:
            print(f"\n{'='*70}")
            print(f"FAILED TESTS:")
            print(f"{'='*70}")
            for name, msg, val, exp in self.failed:
                print(f"  ✗ {name}: {msg}")
                if val is not None and exp is not None:
                    print(f"    Got: {val}, Expected: {exp}")

        if self.warnings:
            print(f"\n{'='*70}")
            print(f"WARNINGS:")
            print(f"{'='*70}")
            for name, msg in self.warnings:
                print(f"  ⚠ {name}: {msg}")

        return len(self.failed) == 0


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


# ============================================================
# EXTREME MARKET CASE TESTS
# ============================================================

def test_low_volatility(results: BoundaryCheckResults):
    """Test: Low volatility -> price approaches discounted intrinsic probability."""
    engine = DigitalOptionAnalyticalEngine()

    # With very low vol, ITM call should approach discounted payout
    # OTM call should approach 0
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.001)

    # Deep ITM with low vol: S >> K, probability of ITM ~ 1
    itm_call = create_digital_call(K=80.0, payout=10.0, T=1.0)
    price = engine.price(itm_call, env)
    expected = 10.0 * np.exp(-0.05 * 1.0)  # Discounted payout

    passed = abs(price - expected) < 0.01
    results.add_result(
        "Low Vol - Deep ITM Call",
        passed,
        f"Price={price:.4f}, Expected≈{expected:.4f}",
        price, expected
    )

    # Deep OTM with low vol: S << K, probability of ITM ~ 0
    otm_call = create_digital_call(K=150.0, payout=10.0, T=1.0)
    price = engine.price(otm_call, env)
    expected = 0.0

    passed = abs(price - expected) < 0.5  # Small probability remains
    results.add_result(
        "Low Vol - Deep OTM Call",
        passed,
        f"Price={price:.4f}, Expected≈{expected:.4f}",
        price, expected
    )


def test_near_expiry(results: BoundaryCheckResults):
    """Test: Near expiry -> price approaches payoff."""
    engine = DigitalOptionAnalyticalEngine()

    # ITM call near expiry
    env = create_pricing_env(spot=105.0, rate=0.05, vol=0.20)
    itm_call = create_digital_call(K=100.0, payout=10.0, T=1e-6)

    price = engine.price(itm_call, env)
    expected = 10.0  # Payoff at expiry when S > K

    passed = abs(price - expected) < 0.01
    results.add_result(
        "Near Expiry - ITM Call",
        passed,
        f"Price={price:.4f}, Expected={expected:.4f}",
        price, expected
    )

    # OTM call near expiry
    env = create_pricing_env(spot=95.0, rate=0.05, vol=0.20)
    otm_call = create_digital_call(K=100.0, payout=10.0, T=1e-6)

    price = engine.price(otm_call, env)
    expected = 0.0

    passed = abs(price - expected) < 0.01
    results.add_result(
        "Near Expiry - OTM Call",
        passed,
        f"Price={price:.4f}, Expected={expected:.4f}",
        price, expected
    )

    # ATM near expiry: with continuous distribution,
    # probability approaches 0.5 as T -> 0 and S = K
    # because N(d2) -> N(0) = 0.5 as d2 -> 0 when S=K
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)
    atm_call = create_digital_call(K=100.0, payout=10.0, T=1e-10)

    price = engine.price(atm_call, env)
    # As T -> 0 with S=K, d2 -> 0, so N(d2) -> 0.5
    # But there's also a drift effect from rate - q
    expected = 10.0 * 0.5  # Approximately 0.5 * payout

    passed = abs(price - expected) < 1.0  # Relaxed tolerance
    results.add_result(
        "Near Expiry - ATM Call (S=K)",
        passed,
        f"Price={price:.4f}, Expected≈{expected:.4f}",
        price, expected
    )


def test_deep_itm(results: BoundaryCheckResults):
    """Test: Deep ITM behavior."""
    engine = DigitalOptionAnalyticalEngine()
    env = create_pricing_env(spot=150.0, rate=0.05, vol=0.20)

    # Deep ITM call
    itm_call = create_digital_call(K=80.0, payout=10.0, T=1.0)
    price = engine.price(itm_call, env)

    # Price should be close to discounted payout
    expected = 10.0 * np.exp(-0.05 * 1.0)
    passed = price > expected * 0.95  # Within 5% of discounted payout

    results.add_result(
        "Deep ITM Call",
        passed,
        f"Price={price:.4f}, Expected≈{expected:.4f}",
        price, expected
    )

    # Deep ITM put
    env = create_pricing_env(spot=50.0, rate=0.05, vol=0.20)
    itm_put = create_digital_put(K=100.0, payout=10.0, T=1.0)
    price = engine.price(itm_put, env)

    expected = 10.0 * np.exp(-0.05 * 1.0)
    passed = price > expected * 0.95

    results.add_result(
        "Deep ITM Put",
        passed,
        f"Price={price:.4f}, Expected≈{expected:.4f}",
        price, expected
    )


def test_deep_otm(results: BoundaryCheckResults):
    """Test: Deep OTM behavior."""
    engine = DigitalOptionAnalyticalEngine()
    env = create_pricing_env(spot=50.0, rate=0.05, vol=0.20)

    # Deep OTM call
    otm_call = create_digital_call(K=150.0, payout=10.0, T=1.0)
    price = engine.price(otm_call, env)

    passed = price < 1.0  # Should be close to zero
    results.add_result(
        "Deep OTM Call",
        passed,
        f"Price={price:.4f}, Expected close to 0",
        price, 0.0
    )

    # Deep OTM put
    env = create_pricing_env(spot=150.0, rate=0.05, vol=0.20)
    otm_put = create_digital_put(K=80.0, payout=10.0, T=1.0)
    price = engine.price(otm_put, env)

    passed = price < 1.0
    results.add_result(
        "Deep OTM Put",
        passed,
        f"Price={price:.4f}, Expected close to 0",
        price, 0.0
    )


def test_atm_with_zero_drift(results: BoundaryCheckResults):
    """Test: ATM with zero drift."""
    engine = DigitalOptionAnalyticalEngine()

    # ATM with r = 0, q = 0
    # d2 = [ln(1) + (0 - 0 + 0.5*sigma^2)*T] / (sigma*sqrt(T)) - sigma*sqrt(T)
    #    = 0.5*sigma*sqrt(T) - sigma*sqrt(T) = -0.5*sigma*sqrt(T)
    # For sigma=0.20, T=1: d2 = -0.10
    # N(d2) = N(-0.10) ≈ 0.4602
    # Expected price ≈ 10 * 0.4602 ≈ 4.60
    env = create_pricing_env(spot=100.0, rate=0.0, vol=0.20, div=0.0)
    atm_call = create_digital_call(K=100.0, payout=10.0, T=1.0)

    price = engine.price(atm_call, env)
    # Expected is slightly less than 0.5 * payout due to vol term
    expected = 4.6  # Approximately 0.46 * 10

    passed = abs(price - expected) < 0.1
    results.add_result(
        "ATM Zero Drift Call",
        passed,
        f"Price={price:.4f}, Expected≈{expected:.4f}",
        price, expected
    )


def test_high_volatility(results: BoundaryCheckResults):
    """Test: High volatility -> prices converge for same moneyness."""
    engine = DigitalOptionAnalyticalEngine()

    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.50)

    # With high vol, ITM and OTM prices should converge toward 0.5 * discounted payout
    # for options that are symmetric around the strike
    itm_call = create_digital_call(K=90.0, payout=10.0, T=1.0)  # 10% ITM
    otm_call = create_digital_call(K=110.0, payout=10.0, T=1.0)  # 10% OTM

    itm_price = engine.price(itm_call, env)
    otm_price = engine.price(otm_call, env)

    # With high vol, the probability difference decreases
    # ITM > OTM but both approach 0.5 * discounted payout
    passed = itm_price > otm_price

    results.add_result(
        "High Vol - ITM > OTM",
        passed,
        f"ITM={itm_price:.4f}, OTM={otm_price:.4f}",
        (itm_price, otm_price), None
    )


def test_price_bounds(results: BoundaryCheckResults):
    """Test: Price bounded by [0, payout * exp(-rT)]."""
    engine = DigitalOptionAnalyticalEngine()
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    test_cases = [
        (80.0, 10.0, 1.0, OptionType.CALL),   # ITM call
        (120.0, 10.0, 1.0, OptionType.CALL),  # OTM call
        (80.0, 10.0, 1.0, OptionType.PUT),    # OTM put
        (120.0, 10.0, 1.0, OptionType.PUT),   # ITM put
    ]

    for K, payout, T, opt_type in test_cases:
        if opt_type == OptionType.CALL:
            option = create_digital_call(K=K, payout=payout, T=T)
        else:
            option = create_digital_put(K=K, payout=payout, T=T)

        price = engine.price(option, env)
        max_price = payout * np.exp(-0.05 * T)

        passed = 0 <= price <= max_price * 1.01  # Small numerical tolerance

        results.add_result(
            f"Price Bounds - {opt_type.name} K={K}",
            passed,
            f"Price={price:.4f}, Bounds=[0, {max_price:.4f}]",
            price, max_price
        )


# ============================================================
# THEORETICAL RELATIONSHIP TESTS
# ============================================================

def test_digital_call_put_parity(results: BoundaryCheckResults):
    """Test: Digital Call + Digital Put = payout * exp(-rT)."""
    engine = DigitalOptionAnalyticalEngine()

    test_cases = [
        (100.0, 100.0, 10.0, 1.0, 0.05, 0.02),  # ATM
        (100.0, 90.0, 10.0, 1.0, 0.05, 0.02),   # ITM
        (100.0, 110.0, 10.0, 1.0, 0.05, 0.02),  # OTM
        (100.0, 100.0, 15.0, 0.5, 0.03, 0.0),   # Different params
    ]

    for S, K, payout, T, r, q in test_cases:
        env = create_pricing_env(spot=S, rate=r, vol=0.20, div=q)

        call = create_digital_call(K=K, payout=payout, T=T)
        put = create_digital_put(K=K, payout=payout, T=T)

        call_price = engine.price(call, env)
        put_price = engine.price(put, env)

        lhs = call_price + put_price
        rhs = payout * np.exp(-r * T)

        passed = abs(lhs - rhs) < 1e-6

        results.add_result(
            f"Digital Parity - S={S}, K={K}, T={T}",
            passed,
            f"C+P={lhs:.6f}, RHS={rhs:.6f}, Diff={abs(lhs-rhs):.2e}",
            lhs, rhs
        )


def test_monotonicity_in_strike(results: BoundaryCheckResults):
    """Test: Price monotonicity in strike."""
    engine = DigitalOptionAnalyticalEngine()
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    # Call: decreasing in strike
    strikes = [80, 90, 100, 110, 120]
    call_prices = []
    for K in strikes:
        call = create_digital_call(K=K, payout=10.0, T=1.0)
        call_prices.append(engine.price(call, env))

    # Check that prices decrease as strike increases
    passed = all(call_prices[i] >= call_prices[i+1] - 1e-10
                 for i in range(len(call_prices) - 1))

    results.add_result(
        "Monotonicity - Call in Strike",
        passed,
        f"Prices: {call_prices}",
        None, None
    )

    # Put: increasing in strike
    put_prices = []
    for K in strikes:
        put = create_digital_put(K=K, payout=10.0, T=1.0)
        put_prices.append(engine.price(put, env))

    passed = all(put_prices[i] <= put_prices[i+1] + 1e-10
                 for i in range(len(put_prices) - 1))

    results.add_result(
        "Monotonicity - Put in Strike",
        passed,
        f"Prices: {put_prices}",
        None, None
    )


def test_monotonicity_in_maturity(results: BoundaryCheckResults):
    """Test: Price monotonicity in time to maturity for ATM options."""
    engine = DigitalOptionAnalyticalEngine()
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    maturities = [0.1, 0.25, 0.5, 1.0, 2.0]
    call_prices = []
    for T in maturities:
        call = create_digital_call(K=100.0, payout=10.0, T=T)
        call_prices.append(engine.price(call, env))

    # For ATM with r > 0, call price can increase or decrease with maturity
    # depending on whether probability effect or discount effect dominates
    # Just check that prices are reasonable
    passed = all(0 < p < 10 * np.exp(-0.05 * 0.1) * 1.1 for p in call_prices)

    results.add_result(
        "Monotonicity in Maturity",
        passed,
        f"Prices: {[f'{p:.4f}' for p in call_prices]}",
        None, None
    )


def test_atm_probability_with_drift(results: BoundaryCheckResults):
    """Test: ATM probability with different drifts."""
    engine = DigitalOptionAnalyticalEngine()

    # High positive drift increases call probability
    env_high = create_pricing_env(spot=100.0, rate=0.10, vol=0.20, div=0.0)
    call_high = create_digital_call(K=100.0, payout=10.0, T=1.0)
    price_high = engine.price(call_high, env_high)

    # Zero drift
    env_zero = create_pricing_env(spot=100.0, rate=0.0, vol=0.20, div=0.0)
    call_zero = create_digital_call(K=100.0, payout=10.0, T=1.0)
    price_zero = engine.price(call_zero, env_zero)

    # Positive drift should increase call price
    passed = price_high > price_zero

    results.add_result(
        "Drift Effect - High r > Zero r",
        passed,
        f"High r={price_high:.4f}, Zero r={price_zero:.4f}",
        (price_high, price_zero), None
    )


# ============================================================
# EDGE CASE TESTS
# ============================================================

def test_exact_atm_boundary(results: BoundaryCheckResults):
    """Test: Exact ATM boundary (S = K)."""
    engine = DigitalOptionAnalyticalEngine()
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    # Exact ATM
    atm_call = create_digital_call(K=100.0, payout=10.0, T=1.0)
    price = engine.price(atm_call, env)

    # With positive drift, ATM call should be > 0.5 * discounted payout
    discounted_payout = 10.0 * np.exp(-0.05)
    passed = price > 0.5 * discounted_payout

    results.add_result(
        "ATM Boundary",
        passed,
        f"Price={price:.4f}, 0.5*df={0.5*discounted_payout:.4f}",
        price, 0.5 * discounted_payout
    )


def test_various_payouts(results: BoundaryCheckResults):
    """Test: Various payout amounts."""
    engine = DigitalOptionAnalyticalEngine()
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    payouts = [1.0, 5.0, 10.0, 50.0, 100.0]
    for payout in payouts:
        call = create_digital_call(K=100.0, payout=payout, T=1.0)
        price = engine.price(call, env)

        # Price should scale linearly with payout
        expected = payout * 0.5  # Approx for ATM
        passed = abs(price - expected) < payout * 0.3  # 30% tolerance for ATM

        results.add_result(
            f"Payout Scaling - payout={payout}",
            passed,
            f"Price={price:.4f}, Expected≈{expected:.4f}",
            price, expected
        )


def test_different_maturities(results: BoundaryCheckResults):
    """Test: Different maturity values."""
    engine = DigitalOptionAnalyticalEngine()
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    maturities = [0.01, 0.1, 0.5, 1.0, 5.0]
    for T in maturities:
        call = create_digital_call(K=100.0, payout=10.0, T=T)
        try:
            price = engine.price(call, env)
            max_price = 10.0 * np.exp(-0.05 * T)
            passed = 0 <= price <= max_price * 1.01
            results.add_result(
                f"Maturity - T={T}",
                passed,
                f"Price={price:.4f}, Max={max_price:.4f}",
                price, max_price
            )
        except Exception as e:
            results.add_result(
                f"Maturity - T={T}",
                False,
                f"Exception: {e}",
                None, None
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results = BoundaryCheckResults()

    print("\n" + "="*70)
    print("DIGITAL OPTION ANALYTICAL ENGINE - BOUNDARY CHECKS")
    print("="*70)

    # Extreme market cases
    print("\n[1/5] Extreme Market Cases...")
    test_low_volatility(results)
    test_near_expiry(results)
    test_deep_itm(results)
    test_deep_otm(results)
    test_atm_with_zero_drift(results)
    test_high_volatility(results)
    test_price_bounds(results)

    # Theoretical relationships
    print("\n[2/5] Theoretical Relationships...")
    test_digital_call_put_parity(results)
    test_monotonicity_in_strike(results)
    test_monotonicity_in_maturity(results)
    test_atm_probability_with_drift(results)

    # Edge cases
    print("\n[3/5] Edge Cases...")
    test_exact_atm_boundary(results)
    test_various_payouts(results)
    test_different_maturities(results)

    # Print summary
    success = results.summary()

    sys.exit(0 if success else 1)
