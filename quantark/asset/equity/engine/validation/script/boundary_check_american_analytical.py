"""
Boundary Check Script for American Option Analytical Engine
Generated: 2025-02-14
"""
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent.parent))

from quantark.asset.equity.product.option import AmericanOption, EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import (
    AmericanOptionAnalyticalEngine,
    BlackScholesEngine,
)
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import AmericanAnalyticalMethod
from quantark.util.numerical import safe_exp


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
        print("BOUNDARY CHECK SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(
            f"Passed: {len(self.passed)} "
            f"({100*len(self.passed)/total if total > 0 else 0:.1f}%)"
        )
        print(
            f"Failed: {len(self.failed)} "
            f"({100*len(self.failed)/total if total > 0 else 0:.1f}%)"
        )
        print(f"Warnings: {len(self.warnings)}")

        if self.failed:
            print("\nFailed Tests:")
            for name, msg in self.failed:
                print(f"  - {name}: {msg}")

        if self.warnings:
            print("\nWarnings:")
            for name, msg in self.warnings:
                print(f"  - {name}: {msg}")

        return len(self.failed) == 0


def create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.0):
    """Helper to create pricing environment."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


# ============================================================
# EXTREME MARKET CASE TESTS
# ============================================================

def test_near_expiry(results: BoundaryCheckResults):
    """Test: Near expiry → payoff."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)
    option = AmericanOption(
        strike=95.0,
        option_type=OptionType.CALL,
        maturity=1e-8,
    )

    engine = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)
    price = engine.price(option, pricing_env)

    intrinsic = max(100.0 - 95.0, 0.0)
    passed = abs(price - intrinsic) < 1.0
    results.add_result(
        "Near expiry (T→0)",
        passed,
        f"Price=${price:.4f}, intrinsic=${intrinsic:.4f}",
    )


def test_low_volatility(results: BoundaryCheckResults):
    """Test: Low volatility → intrinsic value (deep ITM)."""
    pricing_env = create_pricing_env(spot=120.0, rate=0.05, vol=0.001)
    option = AmericanOption(
        strike=90.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )

    engine = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)
    price = engine.price(option, pricing_env)

    rate = pricing_env.get_rate(option.maturity)
    expected = max(120.0 - 90.0 * safe_exp(-rate * option.maturity), 0.0)
    passed = abs(price - expected) < 2.0
    results.add_result(
        "Low volatility (σ→0)",
        passed,
        f"Price=${price:.4f}, expected=${expected:.4f}",
    )


def test_high_volatility(results: BoundaryCheckResults):
    """Test: High volatility → finite, positive price."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=1.0)
    option = AmericanOption(
        strike=100.0,
        option_type=OptionType.PUT,
        maturity=1.0,
    )

    engine = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)
    price = engine.price(option, pricing_env)

    passed = price > 0 and np.isfinite(price)
    results.add_result(
        "High volatility (σ=100%)",
        passed,
        f"Price=${price:.4f}",
    )


def test_deep_otm_call(results: BoundaryCheckResults):
    """Test: Deep OTM call → low value."""
    pricing_env = create_pricing_env(spot=50.0, rate=0.05, vol=0.20)
    option = AmericanOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )

    engine = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)
    price = engine.price(option, pricing_env)

    passed = price >= 0 and price < 2.0
    results.add_result(
        "Deep OTM call",
        passed,
        f"Price=${price:.4f}, expected near 0",
    )


def test_deep_itm_put(results: BoundaryCheckResults):
    """Test: Deep ITM put → high value."""
    pricing_env = create_pricing_env(spot=50.0, rate=0.05, vol=0.20)
    option = AmericanOption(
        strike=120.0,
        option_type=OptionType.PUT,
        maturity=1.0,
    )

    engine = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)
    price = engine.price(option, pricing_env)

    intrinsic = max(120.0 - 50.0, 0.0)
    passed = price >= intrinsic - 1.0
    results.add_result(
        "Deep ITM put",
        passed,
        f"Price=${price:.4f}, intrinsic=${intrinsic:.4f}",
    )


# ============================================================
# THEORETICAL RELATIONSHIP TESTS
# ============================================================

def test_american_ge_european(results: BoundaryCheckResults):
    """Test: American ≥ European (put)."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.02)

    amer_option = AmericanOption(
        strike=100.0,
        option_type=OptionType.PUT,
        maturity=1.0,
    )
    euro_option = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.PUT,
        maturity=1.0,
    )

    amer_engine = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)
    euro_engine = BlackScholesEngine()

    amer_price = amer_engine.price(amer_option, pricing_env)
    euro_price = euro_engine.price(euro_option, pricing_env)

    passed = amer_price >= euro_price - 1e-6
    results.add_result(
        "American ≥ European (put)",
        passed,
        f"American=${amer_price:.4f}, European=${euro_price:.4f}",
    )


def test_call_no_dividend_equals_european(results: BoundaryCheckResults):
    """Test: American call = European call when q=0."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.0)

    amer_option = AmericanOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    euro_option = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )

    amer_engine = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)
    euro_engine = BlackScholesEngine()

    amer_price = amer_engine.price(amer_option, pricing_env)
    euro_price = euro_engine.price(euro_option, pricing_env)

    passed = abs(amer_price - euro_price) < 1e-3
    results.add_result(
        "Call (q=0) equals European",
        passed,
        f"American=${amer_price:.4f}, European=${euro_price:.4f}",
    )


def test_monotonicity_in_strike(results: BoundaryCheckResults):
    """Test: Higher strike → lower call value."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.01)
    strikes = [90.0, 100.0, 110.0]
    prices = []

    engine = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)
    for strike in strikes:
        option = AmericanOption(
            strike=strike,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        prices.append(engine.price(option, pricing_env))

    passed = prices[0] >= prices[1] >= prices[2]
    results.add_result(
        "Call monotonicity in strike",
        passed,
        f"K=90: ${prices[0]:.4f}, K=100: ${prices[1]:.4f}, K=110: ${prices[2]:.4f}",
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results = BoundaryCheckResults()

    test_near_expiry(results)
    test_low_volatility(results)
    test_high_volatility(results)
    test_deep_otm_call(results)
    test_deep_itm_put(results)

    test_american_ge_european(results)
    test_call_no_dividend_equals_european(results)
    test_monotonicity_in_strike(results)

    success = results.summary()
    sys.exit(0 if success else 1)
