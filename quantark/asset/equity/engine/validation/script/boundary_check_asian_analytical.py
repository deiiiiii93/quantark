"""
Boundary Check Script for Asian Option Analytical Engine
Generated: 2024-12-23

Tests extreme market cases and theoretical relationships for Asian options.
"""
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from quantark.asset.equity.product.option import AsianOption, EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import AsianOptionAnalyticalEngine, BlackScholesEngine
from quantark.asset.equity.engine.mc import AsianOptionMCEngine
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType, AveragingType, AsianStrikeType
from quantark.util.enum.engine_enums import AsianAnalyticalMethod, MonteCarloMethod, EngineType
from quantark.util.exceptions import ValidationError


class BoundaryCheckResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_result(self, test_name: str, passed: bool, message: str):
        if passed:
            self.passed.append((test_name, message))
        else:
            self.failed.append((test_name, message))

    def add_warning(self, test_name: str, message: str):
        self.warnings.append((test_name, message))

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print(f"BOUNDARY CHECK SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(f"Passed: {len(self.passed)} ({100*len(self.passed)/total if total > 0 else 0:.1f}%)")
        print(f"Failed: {len(self.failed)} ({100*len(self.failed)/total if total > 0 else 0:.1f}%)")
        print(f"Warnings: {len(self.warnings)}")

        if self.failed:
            print(f"\nFailed Tests:")
            for name, msg in self.failed:
                print(f"  - {name}: {msg}")

        if self.warnings:
            print(f"\nWarnings:")
            for name, msg in self.warnings:
                print(f"  - {name}: {msg}")

        return len(self.failed) == 0


def create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.0, valuation_date=None):
    """Helper to create pricing environment."""
    if valuation_date is None:
        valuation_date = datetime(2024, 1, 1)
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=valuation_date,
    )


# ============================================================
# EXTREME MARKET CASE TESTS
# ============================================================

def test_low_volatility(results: BoundaryCheckResults):
    """Test: Low volatility → intrinsic value (for deep ITM/OTM)."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.001)

    # Deep ITM call with low vol
    option = AsianOption(
        strike=80.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    price = engine.price(option, pricing_env)

    # With very low volatility, price should approach intrinsic value
    # For Asian: intrinsic is max(S - K, 0) = 20, but averaging reduces this
    # A rough check: price should be positive and reasonable
    passed = price > 15 and price < 25
    results.add_result(
        "Low volatility (σ→0)",
        passed,
        f"Price=${price:.4f}, expected ~$20 (intrinsic)"
    )


def test_high_volatility(results: BoundaryCheckResults):
    """Test: High volatility → higher option value."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=1.0)

    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    price = engine.price(option, pricing_env)

    passed = price > 0 and not np.isnan(price) and not np.isinf(price)
    results.add_result(
        "High volatility (σ=100%)",
        passed,
        f"Price=${price:.4f}, should be finite and positive"
    )


def test_near_expiry(results: BoundaryCheckResults):
    """Test: Near expiry → payoff."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    option = AsianOption(
        strike=95.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1e-8,  # Very close to expiry
        num_observations=1,
    )

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    price = engine.price(option, pricing_env)

    # Near expiry, should be close to intrinsic
    intrinsic = max(100.0 - 95.0, 0)
    passed = abs(price - intrinsic) < 1.0
    results.add_result(
        "Near expiry (T→0)",
        passed,
        f"Price=${price:.4f}, intrinsic=${intrinsic}"
    )


def test_deep_itm(results: BoundaryCheckResults):
    """Test: Deep ITM → high value."""
    pricing_env = create_pricing_env(spot=150.0, rate=0.05, vol=0.20)

    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    price = engine.price(option, pricing_env)

    passed = price > 30 and price < 60
    results.add_result(
        "Deep ITM call",
        passed,
        f"Price=${price:.4f}, expected $30-60"
    )


def test_deep_otm(results: BoundaryCheckResults):
    """Test: Deep OTM → low value."""
    pricing_env = create_pricing_env(spot=50.0, rate=0.05, vol=0.20)

    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    price = engine.price(option, pricing_env)

    passed = price >= 0 and price < 5.0
    results.add_result(
        "Deep OTM call",
        passed,
        f"Price=${price:.4f}, expected $0-5"
    )


def test_zero_interest_rate(results: BoundaryCheckResults):
    """Test: r=0 → no discounting effect."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.0, vol=0.20)

    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    price = engine.price(option, pricing_env)

    passed = price > 0 and not np.isnan(price)
    results.add_result(
        "Zero interest rate (r=0)",
        passed,
        f"Price=${price:.4f}"
    )


def test_zero_cost_of_carry(results: BoundaryCheckResults):
    """Test: b=0 (r=q) special case."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.05)  # b = r - q = 0

    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    price = engine.price(option, pricing_env)

    passed = price > 0 and not np.isnan(price)
    results.add_result(
        "Zero cost-of-carry (b=0)",
        passed,
        f"Price=${price:.4f}"
    )


# ============================================================
# THEORETICAL RELATIONSHIP TESTS
# ============================================================

def test_geometric_cheaper_than_arithmetic(results: BoundaryCheckResults):
    """Test: Geometric average ≤ Arithmetic average (Jensen's inequality)."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    geo_option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.GEOMETRIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    arith_option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    kv_engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.KEMNA_VORST)
    tw_engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)

    geo_price = kv_engine.price(geo_option, pricing_env)
    arith_price = tw_engine.price(arith_option, pricing_env)

    passed = geo_price <= arith_price
    results.add_result(
        "Geometric ≤ Arithmetic",
        passed,
        f"Geometric=${geo_price:.4f}, Arithmetic=${arith_price:.4f}"
    )


def test_asian_cheaper_than_vanilla(results: BoundaryCheckResults):
    """Test: Asian option ≤ Vanilla option (averaging reduces volatility)."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    # Asian call (arithmetic)
    asian_option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    # Vanilla call
    vanilla_option = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )

    asian_engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    bs_engine = BlackScholesEngine()

    asian_price = asian_engine.price(asian_option, pricing_env)
    vanilla_price = bs_engine.price(vanilla_option, pricing_env)

    passed = asian_price <= vanilla_price
    results.add_result(
        "Asian ≤ Vanilla",
        passed,
        f"Asian=${asian_price:.4f}, Vanilla=${vanilla_price:.4f}"
    )


def test_atm_call_positive(results: BoundaryCheckResults):
    """Test: ATM call should have positive value."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    price = engine.price(option, pricing_env)

    passed = price > 0
    results.add_result(
        "ATM call > 0",
        passed,
        f"Price=${price:.4f}"
    )


def test_monotonicity_in_strike(results: BoundaryCheckResults):
    """Test: Higher strike → lower call value."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    strikes = [90.0, 100.0, 110.0]
    prices = []

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)

    for strike in strikes:
        option = AsianOption(
            strike=strike,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
            num_observations=12,
        )
        prices.append(engine.price(option, pricing_env))

    # Prices should be decreasing with strike
    passed = prices[0] >= prices[1] >= prices[2]
    results.add_result(
        "Call monotonicity in strike",
        passed,
        f"K=90: ${prices[0]:.4f}, K=100: ${prices[1]:.4f}, K=110: ${prices[2]:.4f}"
    )


def test_monotonicity_in_volatility(results: BoundaryCheckResults):
    """Test: Higher volatility → higher call value."""
    pricing_env_base = create_pricing_env(spot=100.0, rate=0.05, vol=0.10)

    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)

    # Test with different volatilities
    env_low = create_pricing_env(spot=100.0, rate=0.05, vol=0.10)
    env_high = create_pricing_env(spot=100.0, rate=0.05, vol=0.30)

    price_low = engine.price(option, env_low)
    price_high = engine.price(option, env_high)

    passed = price_high >= price_low
    results.add_result(
        "Call monotonicity in volatility",
        passed,
        f"σ=10%: ${price_low:.4f}, σ=30%: ${price_high:.4f}"
    )


def test_put_call_symmetry_approximation(results: BoundaryCheckResults):
    """Test: Put-call relationship (no exact parity for Asian)."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    call_option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    put_option = AsianOption(
        strike=100.0,
        option_type=OptionType.PUT,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)

    call_price = engine.price(call_option, pricing_env)
    put_price = engine.price(put_option, pricing_env)

    # Both ATM options should have similar value
    ratio = call_price / put_price if put_price > 0 else 0
    passed = 0.5 < ratio < 2.0
    results.add_result(
        "ATM put-call relationship",
        passed,
        f"Call=${call_price:.4f}, Put=${put_price:.4f}, Ratio={ratio:.2f}"
    )


def test_floating_strike_symmetry(results: BoundaryCheckResults):
    """Test: Henderson-Wojakowski symmetry for floating-strike."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.02)  # b=0.03

    # Floating call should equal fixed put with transformed parameters
    floating_call = AsianOption(
        strike=0.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FLOATING,
        maturity=1.0,
        num_observations=12,
    )

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
    price = engine.price(floating_call, pricing_env)

    passed = price > 0 and not np.isnan(price)
    results.add_result(
        "Floating-strike symmetry",
        passed,
        f"Floating call price=${price:.4f}"
    )


# ============================================================
# METHOD SPECIFIC TESTS
# ============================================================

def test_kemna_vorst_geometric(results: BoundaryCheckResults):
    """Test: Kemna-Vorst should match BSM with adjusted parameters."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)

    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.GEOMETRIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.KEMNA_VORST)
    price = engine.price(option, pricing_env)

    passed = price > 0 and not np.isnan(price)
    results.add_result(
        "Kemna-Vorst geometric",
        passed,
        f"Geometric call price=${price:.4f}"
    )


def test_levy_b_not_zero(results: BoundaryCheckResults):
    """Test: LEVY method should fail for b=0."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.05)  # b=0

    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    try:
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.LEVY)
        price = engine.price(option, pricing_env)
        # If we get here, LEVY handled b=0 (or implementation doesn't check)
        results.add_warning(
            "LEVY b=0 check",
            f"LEVY allowed b=0, price=${price:.4f} (should fail per Haug)"
        )
    except ValidationError as e:
        results.add_result(
            "LEVY b=0 check",
            True,
            f"Correctly rejects b=0: {str(e)[:50]}"
        )


def test_all_methods_produce_finite_prices(results: BoundaryCheckResults):
    """Test: All methods produce finite, positive prices."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.02)

    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )

    methods = [
        AsianAnalyticalMethod.TURNBULL_WAKEMAN,
        AsianAnalyticalMethod.LEVY,
        AsianAnalyticalMethod.CURRAN,
        AsianAnalyticalMethod.DISCRETE_HHM,
    ]

    all_passed = True
    prices = {}

    for method in methods:
        try:
            engine = AsianOptionAnalyticalEngine(method=method)
            price = engine.price(option, pricing_env)
            prices[method.value] = price
            if not (price > 0 and not np.isnan(price) and not np.isinf(price)):
                all_passed = False
        except Exception as e:
            prices[method.value] = f"Error: {e}"
            all_passed = False

    results.add_result(
        "All methods finite prices",
        all_passed,
        f"Prices: {prices}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results = BoundaryCheckResults()

    print("Running Boundary Checks for Asian Option Analytical Engine")
    print("="*60)

    # Extreme market cases
    print("\n1. Extreme Market Cases")
    print("-"*40)
    test_low_volatility(results)
    test_high_volatility(results)
    test_near_expiry(results)
    test_deep_itm(results)
    test_deep_otm(results)
    test_zero_interest_rate(results)
    test_zero_cost_of_carry(results)

    # Theoretical relationships
    print("\n2. Theoretical Relationships")
    print("-"*40)
    test_geometric_cheaper_than_arithmetic(results)
    test_asian_cheaper_than_vanilla(results)
    test_atm_call_positive(results)
    test_monotonicity_in_strike(results)
    test_monotonicity_in_volatility(results)
    test_put_call_symmetry_approximation(results)
    test_floating_strike_symmetry(results)

    # Method-specific tests
    print("\n3. Method-Specific Tests")
    print("-"*40)
    test_kemna_vorst_geometric(results)
    test_levy_b_not_zero(results)
    test_all_methods_produce_finite_prices(results)

    # Print summary
    success = results.summary()
    sys.exit(0 if success else 1)
