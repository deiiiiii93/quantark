"""
Boundary Check Template for Engine Validation
==============================================

This template provides a comprehensive boundary check framework for pricing engines.
Copy and customize for specific engine validation.

Usage:
    1. Copy this file to: asset/<type>/engine/validation/script/boundary_check_<engine_name>.py
    2. Replace placeholders with actual imports and implementations
    3. Add product-specific boundary checks as needed
    4. Run: python asset/<type>/engine/validation/script/boundary_check_<engine_name>.py
"""
import numpy as np
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

# Add project root to path
sys.path.insert(0, '.')

# ==============================================================================
# REPLACE THESE IMPORTS WITH ACTUAL ENGINE/PRODUCT IMPORTS
# ==============================================================================
# from asset.equity.product.option.<product> import <Product>
# from asset.equity.engine.analytical.<engine> import <Engine>
from priceenv.pricing_environment import PricingEnvironment
from param.spot_quote import SpotQuote
from param.rate_curve import FlatRateCurve
from param.vol_surface import FlatVolSurface
from param.dividend import ContinuousDividendYield


class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARN = "WARN"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    message: str
    expected: Optional[float] = None
    actual: Optional[float] = None
    tolerance: Optional[float] = None


class BoundaryCheckResults:
    """Collects and reports boundary check results."""
    
    def __init__(self):
        self.results: List[TestResult] = []
    
    def add_pass(self, name: str, message: str, expected: float = None, actual: float = None):
        self.results.append(TestResult(name, TestStatus.PASS, message, expected, actual))
    
    def add_fail(self, name: str, message: str, expected: float = None, actual: float = None, tolerance: float = None):
        self.results.append(TestResult(name, TestStatus.FAIL, message, expected, actual, tolerance))
    
    def add_warn(self, name: str, message: str):
        self.results.append(TestResult(name, TestStatus.WARN, message))
    
    def add_skip(self, name: str, message: str):
        self.results.append(TestResult(name, TestStatus.SKIP, message))
    
    @property
    def passed(self) -> List[TestResult]:
        return [r for r in self.results if r.status == TestStatus.PASS]
    
    @property
    def failed(self) -> List[TestResult]:
        return [r for r in self.results if r.status == TestStatus.FAIL]
    
    @property
    def warnings(self) -> List[TestResult]:
        return [r for r in self.results if r.status == TestStatus.WARN]
    
    @property
    def skipped(self) -> List[TestResult]:
        return [r for r in self.results if r.status == TestStatus.SKIP]
    
    def summary(self) -> bool:
        """Print summary and return True if all tests passed."""
        total = len(self.results)
        n_pass = len(self.passed)
        n_fail = len(self.failed)
        n_warn = len(self.warnings)
        n_skip = len(self.skipped)
        
        print(f"\n{'='*70}")
        print(f"BOUNDARY CHECK SUMMARY")
        print(f"{'='*70}")
        print(f"Total Tests: {total}")
        print(f"  Passed:   {n_pass:3d} ({100*n_pass/total:.1f}%)" if total > 0 else "  Passed:   0")
        print(f"  Failed:   {n_fail:3d} ({100*n_fail/total:.1f}%)" if total > 0 else "  Failed:   0")
        print(f"  Warnings: {n_warn:3d}")
        print(f"  Skipped:  {n_skip:3d}")
        
        if self.failed:
            print(f"\n{'='*70}")
            print("FAILED TESTS:")
            print(f"{'='*70}")
            for r in self.failed:
                print(f"\n  [{r.name}]")
                print(f"    {r.message}")
                if r.expected is not None and r.actual is not None:
                    print(f"    Expected: {r.expected:.6f}, Actual: {r.actual:.6f}")
                    if r.tolerance:
                        print(f"    Tolerance: {r.tolerance:.6f}")
        
        if self.warnings:
            print(f"\n{'='*70}")
            print("WARNINGS:")
            print(f"{'='*70}")
            for r in self.warnings:
                print(f"  [{r.name}] {r.message}")
        
        print(f"\n{'='*70}")
        overall = "PASSED" if n_fail == 0 else "FAILED"
        print(f"OVERALL STATUS: {overall}")
        print(f"{'='*70}\n")
        
        return n_fail == 0


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_pricing_env(spot: float, rate: float, vol: float, div: float = 0.0) -> PricingEnvironment:
    """Create a pricing environment with flat parameters."""
    return PricingEnvironment(
        spot=SpotQuote(spot),
        rate_curve=FlatRateCurve(rate),
        vol_surface=FlatVolSurface(vol),
        dividend=ContinuousDividendYield(div)
    )


def is_close(a: float, b: float, rel_tol: float = 1e-4, abs_tol: float = 1e-8) -> bool:
    """Check if two floats are approximately equal."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# ==============================================================================
# EXTREME MARKET CASE TESTS
# ==============================================================================

def test_low_volatility(results: BoundaryCheckResults, Engine, Product):
    """
    Test: Very low volatility should give intrinsic value.
    
    For European call: max(S*e^(-qT) - K*e^(-rT), 0) approximately
    For European put: max(K*e^(-rT) - S*e^(-qT), 0) approximately
    """
    test_name = "Low Volatility → Intrinsic Value"
    
    try:
        spot, strike, rate, div, maturity = 100.0, 100.0, 0.05, 0.02, 1.0
        vol = 0.001  # Very low volatility
        
        env = create_pricing_env(spot, rate, vol, div)
        engine = Engine()
        
        # Call option
        call = Product(strike=strike, maturity=maturity, is_call=True)
        call_price = engine.price(call, env)
        
        # Expected: forward - strike, discounted (if positive)
        forward = spot * np.exp(-div * maturity)
        strike_pv = strike * np.exp(-rate * maturity)
        expected_call = max(forward - strike_pv, 0)
        
        if is_close(call_price, expected_call, rel_tol=0.01):
            results.add_pass(test_name + " (Call)", 
                           f"Call price {call_price:.4f} ≈ intrinsic {expected_call:.4f}")
        else:
            results.add_fail(test_name + " (Call)",
                           f"Call price differs from intrinsic value",
                           expected_call, call_price, 0.01)
        
        # Put option
        put = Product(strike=strike, maturity=maturity, is_call=False)
        put_price = engine.price(put, env)
        expected_put = max(strike_pv - forward, 0)
        
        if is_close(put_price, expected_put, rel_tol=0.01):
            results.add_pass(test_name + " (Put)",
                           f"Put price {put_price:.4f} ≈ intrinsic {expected_put:.4f}")
        else:
            results.add_fail(test_name + " (Put)",
                           f"Put price differs from intrinsic value",
                           expected_put, put_price, 0.01)
    
    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_near_expiry(results: BoundaryCheckResults, Engine, Product):
    """
    Test: Near expiry should give payoff value.
    
    As T → 0, option value → max(S - K, 0) for call, max(K - S, 0) for put
    """
    test_name = "Near Expiry → Payoff"
    
    try:
        spot, strike, rate, vol, div = 100.0, 100.0, 0.05, 0.2, 0.02
        maturity = 1e-6  # Very near expiry
        
        env = create_pricing_env(spot, rate, vol, div)
        engine = Engine()
        
        # Test ITM call
        itm_call = Product(strike=95.0, maturity=maturity, is_call=True)
        itm_call_price = engine.price(itm_call, env)
        expected_call = max(spot - 95.0, 0)
        
        if is_close(itm_call_price, expected_call, rel_tol=0.01, abs_tol=0.01):
            results.add_pass(test_name + " (ITM Call)",
                           f"ITM call {itm_call_price:.4f} ≈ payoff {expected_call:.4f}")
        else:
            results.add_fail(test_name + " (ITM Call)",
                           f"ITM call differs from payoff",
                           expected_call, itm_call_price)
        
        # Test ITM put
        itm_put = Product(strike=105.0, maturity=maturity, is_call=False)
        itm_put_price = engine.price(itm_put, env)
        expected_put = max(105.0 - spot, 0)
        
        if is_close(itm_put_price, expected_put, rel_tol=0.01, abs_tol=0.01):
            results.add_pass(test_name + " (ITM Put)",
                           f"ITM put {itm_put_price:.4f} ≈ payoff {expected_put:.4f}")
        else:
            results.add_fail(test_name + " (ITM Put)",
                           f"ITM put differs from payoff",
                           expected_put, itm_put_price)
    
    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_deep_itm(results: BoundaryCheckResults, Engine, Product):
    """
    Test: Deep ITM options should have delta close to ±1.
    
    Deep ITM call: delta ≈ 1
    Deep ITM put: delta ≈ -1
    """
    test_name = "Deep ITM → Delta ≈ ±1"
    
    try:
        spot, rate, vol, div, maturity = 100.0, 0.05, 0.2, 0.02, 1.0
        
        env = create_pricing_env(spot, rate, vol, div)
        engine = Engine()
        
        # Deep ITM call (strike = 50)
        deep_itm_call = Product(strike=50.0, maturity=maturity, is_call=True)
        call_price = engine.price(deep_itm_call, env)
        
        # Compute numerical delta
        bump = 0.01
        env_up = create_pricing_env(spot * (1 + bump), rate, vol, div)
        call_price_up = engine.price(deep_itm_call, env_up)
        delta_call = (call_price_up - call_price) / (spot * bump)
        
        if delta_call > 0.95:
            results.add_pass(test_name + " (Call)",
                           f"Deep ITM call delta = {delta_call:.4f} ≈ 1")
        else:
            results.add_fail(test_name + " (Call)",
                           f"Deep ITM call delta should be close to 1",
                           1.0, delta_call)
        
        # Deep ITM put (strike = 150)
        deep_itm_put = Product(strike=150.0, maturity=maturity, is_call=False)
        put_price = engine.price(deep_itm_put, env)
        put_price_up = engine.price(deep_itm_put, env_up)
        delta_put = (put_price_up - put_price) / (spot * bump)
        
        if delta_put < -0.95:
            results.add_pass(test_name + " (Put)",
                           f"Deep ITM put delta = {delta_put:.4f} ≈ -1")
        else:
            results.add_fail(test_name + " (Put)",
                           f"Deep ITM put delta should be close to -1",
                           -1.0, delta_put)
    
    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_deep_otm(results: BoundaryCheckResults, Engine, Product):
    """
    Test: Deep OTM options should have near-zero value.
    """
    test_name = "Deep OTM → Near Zero Value"
    
    try:
        spot, rate, vol, div, maturity = 100.0, 0.05, 0.2, 0.02, 1.0
        
        env = create_pricing_env(spot, rate, vol, div)
        engine = Engine()
        
        # Deep OTM call (strike = 200)
        deep_otm_call = Product(strike=200.0, maturity=maturity, is_call=True)
        call_price = engine.price(deep_otm_call, env)
        
        if call_price < 0.01:
            results.add_pass(test_name + " (Call)",
                           f"Deep OTM call = {call_price:.6f} ≈ 0")
        else:
            results.add_warn(test_name + " (Call)",
                           f"Deep OTM call = {call_price:.6f}, expected near 0")
        
        # Deep OTM put (strike = 50)
        deep_otm_put = Product(strike=50.0, maturity=maturity, is_call=False)
        put_price = engine.price(deep_otm_put, env)
        
        if put_price < 0.01:
            results.add_pass(test_name + " (Put)",
                           f"Deep OTM put = {put_price:.6f} ≈ 0")
        else:
            results.add_warn(test_name + " (Put)",
                           f"Deep OTM put = {put_price:.6f}, expected near 0")
    
    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


# ==============================================================================
# THEORETICAL RELATIONSHIP TESTS
# ==============================================================================

def test_put_call_parity(results: BoundaryCheckResults, Engine, Product):
    """
    Test: Put-Call Parity should hold.
    
    C - P = S*e^(-qT) - K*e^(-rT)
    """
    test_name = "Put-Call Parity"
    
    try:
        spot, strike, rate, vol, div, maturity = 100.0, 100.0, 0.05, 0.2, 0.02, 1.0
        
        env = create_pricing_env(spot, rate, vol, div)
        engine = Engine()
        
        call = Product(strike=strike, maturity=maturity, is_call=True)
        put = Product(strike=strike, maturity=maturity, is_call=False)
        
        call_price = engine.price(call, env)
        put_price = engine.price(put, env)
        
        # C - P
        lhs = call_price - put_price
        
        # S*e^(-qT) - K*e^(-rT)
        rhs = spot * np.exp(-div * maturity) - strike * np.exp(-rate * maturity)
        
        if is_close(lhs, rhs, rel_tol=1e-4):
            results.add_pass(test_name,
                           f"C - P = {lhs:.6f}, S*exp(-qT) - K*exp(-rT) = {rhs:.6f}")
        else:
            results.add_fail(test_name,
                           f"Put-Call Parity violated",
                           rhs, lhs, 1e-4)
    
    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_call_spread_monotonicity(results: BoundaryCheckResults, Engine, Product):
    """
    Test: Call price decreases as strike increases.
    
    C(K1) >= C(K2) when K1 < K2
    """
    test_name = "Call Spread Monotonicity"
    
    try:
        spot, rate, vol, div, maturity = 100.0, 0.05, 0.2, 0.02, 1.0
        
        env = create_pricing_env(spot, rate, vol, div)
        engine = Engine()
        
        strikes = [90, 95, 100, 105, 110]
        prices = []
        
        for k in strikes:
            call = Product(strike=float(k), maturity=maturity, is_call=True)
            prices.append(engine.price(call, env))
        
        # Check monotonicity
        is_monotonic = all(prices[i] >= prices[i+1] for i in range(len(prices)-1))
        
        if is_monotonic:
            results.add_pass(test_name,
                           f"Call prices decrease with strike: {[f'{p:.4f}' for p in prices]}")
        else:
            results.add_fail(test_name,
                           f"Call prices not monotonic: {[f'{p:.4f}' for p in prices]}")
    
    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_butterfly_spread(results: BoundaryCheckResults, Engine, Product):
    """
    Test: Butterfly spread should be non-negative (convexity).
    
    C(K1) + C(K3) >= 2*C(K2) where K1 < K2 < K3 and K2 - K1 = K3 - K2
    """
    test_name = "Butterfly Spread (Convexity)"
    
    try:
        spot, rate, vol, div, maturity = 100.0, 0.05, 0.2, 0.02, 1.0
        
        env = create_pricing_env(spot, rate, vol, div)
        engine = Engine()
        
        k1, k2, k3 = 95.0, 100.0, 105.0
        
        c1 = engine.price(Product(strike=k1, maturity=maturity, is_call=True), env)
        c2 = engine.price(Product(strike=k2, maturity=maturity, is_call=True), env)
        c3 = engine.price(Product(strike=k3, maturity=maturity, is_call=True), env)
        
        butterfly = c1 + c3 - 2 * c2
        
        if butterfly >= -1e-6:  # Allow small numerical tolerance
            results.add_pass(test_name,
                           f"Butterfly = {butterfly:.6f} >= 0")
        else:
            results.add_fail(test_name,
                           f"Butterfly spread negative (convexity violated)",
                           0.0, butterfly)
    
    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_gamma_positive(results: BoundaryCheckResults, Engine, Product):
    """
    Test: Gamma should be non-negative.
    """
    test_name = "Gamma >= 0"
    
    try:
        spot, strike, rate, vol, div, maturity = 100.0, 100.0, 0.05, 0.2, 0.02, 1.0
        
        engine = Engine()
        call = Product(strike=strike, maturity=maturity, is_call=True)
        
        # Compute gamma numerically
        bump = 0.01
        env_base = create_pricing_env(spot, rate, vol, div)
        env_up = create_pricing_env(spot * (1 + bump), rate, vol, div)
        env_down = create_pricing_env(spot * (1 - bump), rate, vol, div)
        
        p_base = engine.price(call, env_base)
        p_up = engine.price(call, env_up)
        p_down = engine.price(call, env_down)
        
        gamma = (p_up - 2*p_base + p_down) / (spot * bump) ** 2
        
        if gamma >= -1e-6:
            results.add_pass(test_name,
                           f"Gamma = {gamma:.6f} >= 0")
        else:
            results.add_fail(test_name,
                           f"Gamma is negative",
                           0.0, gamma)
    
    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_vega_positive(results: BoundaryCheckResults, Engine, Product):
    """
    Test: Vega should be non-negative for vanilla options.
    """
    test_name = "Vega >= 0"
    
    try:
        spot, strike, rate, vol, div, maturity = 100.0, 100.0, 0.05, 0.2, 0.02, 1.0
        
        engine = Engine()
        call = Product(strike=strike, maturity=maturity, is_call=True)
        
        # Compute vega numerically
        bump = 0.01
        env_base = create_pricing_env(spot, rate, vol, div)
        env_up = create_pricing_env(spot, rate, vol + bump, div)
        
        p_base = engine.price(call, env_base)
        p_up = engine.price(call, env_up)
        
        vega = (p_up - p_base) / bump
        
        if vega >= -1e-6:
            results.add_pass(test_name,
                           f"Vega = {vega:.4f} >= 0")
        else:
            results.add_fail(test_name,
                           f"Vega is negative",
                           0.0, vega)
    
    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


# ==============================================================================
# PRODUCT-SPECIFIC TESTS (ADD AS NEEDED)
# ==============================================================================

def test_barrier_specific_ko_ki_sum(results: BoundaryCheckResults, KOEngine, KIEngine, VanillaEngine, BarrierProduct, VanillaProduct):
    """
    Test: KO + KI = Vanilla (for same barrier, no rebate)
    
    Only applicable for barrier options.
    """
    test_name = "KO + KI = Vanilla"
    
    try:
        spot, strike, rate, vol, div, maturity = 100.0, 100.0, 0.05, 0.2, 0.02, 1.0
        barrier = 120.0
        
        env = create_pricing_env(spot, rate, vol, div)
        
        ko = BarrierProduct(strike=strike, barrier=barrier, maturity=maturity, 
                           is_call=True, barrier_type='up-and-out', rebate=0.0)
        ki = BarrierProduct(strike=strike, barrier=barrier, maturity=maturity,
                           is_call=True, barrier_type='up-and-in', rebate=0.0)
        vanilla = VanillaProduct(strike=strike, maturity=maturity, is_call=True)
        
        ko_price = KOEngine().price(ko, env)
        ki_price = KIEngine().price(ki, env)
        vanilla_price = VanillaEngine().price(vanilla, env)
        
        if is_close(ko_price + ki_price, vanilla_price, rel_tol=1e-4):
            results.add_pass(test_name,
                           f"KO({ko_price:.4f}) + KI({ki_price:.4f}) = {ko_price+ki_price:.4f} ≈ Vanilla({vanilla_price:.4f})")
        else:
            results.add_fail(test_name,
                           f"KO + KI != Vanilla",
                           vanilla_price, ko_price + ki_price)
    
    except Exception as e:
        results.add_skip(test_name, f"Not applicable or error: {str(e)}")


# ==============================================================================
# MAIN
# ==============================================================================

def run_all_tests(Engine, Product):
    """Run all boundary check tests."""
    results = BoundaryCheckResults()
    
    print("Running Boundary Checks...")
    print("=" * 70)
    
    # Extreme market cases
    print("\n[Extreme Market Cases]")
    test_low_volatility(results, Engine, Product)
    test_near_expiry(results, Engine, Product)
    test_deep_itm(results, Engine, Product)
    test_deep_otm(results, Engine, Product)
    
    # Theoretical relationships
    print("\n[Theoretical Relationships]")
    test_put_call_parity(results, Engine, Product)
    test_call_spread_monotonicity(results, Engine, Product)
    test_butterfly_spread(results, Engine, Product)
    test_gamma_positive(results, Engine, Product)
    test_vega_positive(results, Engine, Product)
    
    # Print summary
    success = results.summary()
    
    return success, results


if __name__ == "__main__":
    # REPLACE WITH ACTUAL IMPORTS
    # from asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
    # from asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
    # success, results = run_all_tests(BlackScholesEngine, EuropeanVanillaOption)
    
    print("This is a template file.")
    print("Please copy and customize for your specific engine validation.")
    print("\nUsage:")
    print("  1. Copy to: asset/<type>/engine/validation/script/boundary_check_<engine>.py")
    print("  2. Update imports for your engine and product")
    print("  3. Run: python <path_to_script>")
    
    sys.exit(0)
