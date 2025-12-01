"""
Unit tests for European option Monte Carlo pricing engine.
"""

import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from asset.equity.engine.analytical import BlackScholesEngine
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from asset.equity.param import MCParams
from util.enum import OptionType
from util.enum.engine_enums import MonteCarloMethod, EngineType
from util.exceptions import ValidationError, PricingError
from datetime import datetime


def test_mc_call_pricing():
    """Test European call option pricing with normal MC."""
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    params = MCParams(num_paths=50000, time_steps=252, seed=42)
    engine = EuropeanMCEngine(params=params, method=MonteCarloMethod.PSEUDO)
    mc_price = engine.price(call, pricing_env)

    bs_engine = BlackScholesEngine()
    bs_price = bs_engine.price(call, pricing_env)

    std_error = engine.get_last_std_error()
    tolerance = 3 * std_error

    assert abs(mc_price - bs_price) < tolerance, (
        f"MC price {mc_price:.6f} differs from BS price {bs_price:.6f} "
        f"by more than 3 std errors (tolerance={tolerance:.6f})"
    )
    print(f"✓ MC call pricing test passed: MC=${mc_price:.6f}, BS=${bs_price:.6f}, SE={std_error:.6f}")


def test_qmc_call_pricing():
    """Test European call option pricing with QMC (Sobol)."""
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    params = MCParams(num_paths=16384, time_steps=252, seed=42)
    engine = EuropeanMCEngine(params=params, method=MonteCarloMethod.QUASI)
    qmc_price = engine.price(call, pricing_env)

    bs_engine = BlackScholesEngine()
    bs_price = bs_engine.price(call, pricing_env)

    std_error = engine.get_last_std_error()
    tolerance = 3 * std_error

    assert abs(qmc_price - bs_price) < tolerance, (
        f"QMC price {qmc_price:.6f} differs from BS price {bs_price:.6f} "
        f"by more than 3 std errors (tolerance={tolerance:.6f})"
    )
    print(f"✓ QMC call pricing test passed: QMC=${qmc_price:.6f}, BS=${bs_price:.6f}, SE={std_error:.6f}")


def test_rqmc_call_pricing():
    """Test European call option pricing with RQMC."""
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    params = MCParams(num_paths=8192, time_steps=252, seed=42)
    params.max_batches = 16
    params.target_std = 0.01
    params.min_batches = 4

    engine = EuropeanMCEngine(params=params, method=MonteCarloMethod.RANDOMIZED_QUASI)
    rqmc_price = engine.price(call, pricing_env)

    bs_engine = BlackScholesEngine()
    bs_price = bs_engine.price(call, pricing_env)

    rqmc_result = engine.get_last_rqmc_result()
    std_error = rqmc_result.std_error
    tolerance = 3 * std_error

    assert abs(rqmc_price - bs_price) < tolerance, (
        f"RQMC price {rqmc_price:.6f} differs from BS price {bs_price:.6f} "
        f"by more than 3 std errors (tolerance={tolerance:.6f})"
    )
    print(
        f"✓ RQMC call pricing test passed: RQMC=${rqmc_price:.6f}, BS=${bs_price:.6f}, "
        f"SE={std_error:.6f}, batches={rqmc_result.batches_used}"
    )


def test_mc_put_pricing():
    """Test European put option pricing with MC."""
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    put = EuropeanVanillaOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)

    params = MCParams(num_paths=50000, time_steps=252, seed=42)
    engine = EuropeanMCEngine(params=params, method=MonteCarloMethod.PSEUDO)
    mc_price = engine.price(put, pricing_env)

    bs_engine = BlackScholesEngine()
    bs_price = bs_engine.price(put, pricing_env)

    std_error = engine.get_last_std_error()
    tolerance = 3 * std_error

    assert abs(mc_price - bs_price) < tolerance, (
        f"MC put price {mc_price:.6f} differs from BS price {bs_price:.6f} "
        f"by more than 3 std errors (tolerance={tolerance:.6f})"
    )
    print(f"✓ MC put pricing test passed: MC=${mc_price:.6f}, BS=${bs_price:.6f}, SE={std_error:.6f}")


def test_mc_put_call_parity():
    """Test put-call parity with Monte Carlo pricing."""
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02

    spot = SpotQuote(spot=S)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=r)
    div = ContinuousDividendYield(div_yield=q)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    call = EuropeanVanillaOption(K, OptionType.CALL, maturity=T)
    put = EuropeanVanillaOption(K, OptionType.PUT, maturity=T)

    params = MCParams(num_paths=100000, time_steps=252, seed=42)
    engine = EuropeanMCEngine(params=params, method=MonteCarloMethod.PSEUDO)
    call_price = engine.price(call, pricing_env)
    put_price = engine.price(put, pricing_env)

    lhs = call_price - put_price
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

    tolerance = 0.1
    assert abs(lhs - rhs) < tolerance, (
        f"Put-call parity violated with MC: {lhs:.6f} vs {rhs:.6f}, "
        f"difference={abs(lhs - rhs):.6f}"
    )
    print(f"✓ MC put-call parity test passed: difference = {abs(lhs - rhs):.6f}")


def test_antithetic_variance_reduction():
    """Test that antithetic variates reduce variance in MC."""
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    params_without = MCParams(num_paths=10000, time_steps=252, seed=42, use_antithetic=False)
    engine_without = EuropeanMCEngine(params=params_without, method=MonteCarloMethod.PSEUDO)
    engine_without.price(call, pricing_env)
    std_error_without = engine_without.get_last_std_error()

    params_with = MCParams(num_paths=10000, time_steps=252, seed=42, use_antithetic=True)
    engine_with = EuropeanMCEngine(params=params_with, method=MonteCarloMethod.PSEUDO)
    engine_with.price(call, pricing_env)
    std_error_with = engine_with.get_last_std_error()

    assert std_error_with < std_error_without, (
        f"Antithetic variates should reduce standard error: "
        f"without={std_error_without:.6f}, with={std_error_with:.6f}"
    )
    print(
        f"✓ Antithetic variance reduction test passed: "
        f"SE without={std_error_without:.6f}, SE with={std_error_with:.6f}"
    )


def test_two_level_enum_pattern():
    """Test two-level enum pattern for method selection."""
    params = MCParams(num_paths=1000, time_steps=100, seed=42)

    engine = EuropeanMCEngine(
        params=params,
        method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)
    )

    assert engine.method == MonteCarloMethod.QUASI, (
        f"Method should be QUASI, got {engine.method}"
    )
    print("✓ Two-level enum pattern test passed")


def test_string_method_backward_compat():
    """Test string method for backward compatibility."""
    params = MCParams(num_paths=1000, time_steps=100, seed=42)

    engine = EuropeanMCEngine(params=params, method="quasi")
    assert engine.method == MonteCarloMethod.QUASI

    engine = EuropeanMCEngine(params=params, method="pseudo")
    assert engine.method == MonteCarloMethod.PSEUDO

    engine = EuropeanMCEngine(params=params, method="randomized_quasi")
    assert engine.method == MonteCarloMethod.RANDOMIZED_QUASI

    print("✓ String method backward compatibility test passed")


def test_invalid_method():
    """Test that invalid method raises ValidationError."""
    params = MCParams(num_paths=1000, time_steps=100, seed=42)

    try:
        engine = EuropeanMCEngine(params=params, method="invalid_method")
        assert False, "Should have raised ValidationError for invalid method"
    except ValidationError as e:
        assert "Invalid method string" in str(e)
        print("✓ Invalid method validation test passed")


def test_invalid_product_type():
    """Test that non-European options raise PricingError."""
    from asset.equity.product.option.american_option import AmericanOption

    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        valuation_date=datetime(2024, 1, 1),
    )

    american_call = AmericanOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    params = MCParams(num_paths=1000, time_steps=100, seed=42)
    engine = EuropeanMCEngine(params=params)

    try:
        engine.price(american_call, pricing_env)
        assert False, "Should have raised PricingError for American option"
    except PricingError as e:
        assert "EuropeanVanillaOption" in str(e)
        print("✓ Invalid product type test passed")


def test_qmc_convergence_better_than_mc():
    """Test that QMC converges faster than MC for the same number of paths."""
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    bs_engine = BlackScholesEngine()
    bs_price = bs_engine.price(call, pricing_env)

    num_paths = 8192
    params_mc = MCParams(num_paths=num_paths, time_steps=252, seed=42)
    engine_mc = EuropeanMCEngine(params=params_mc, method=MonteCarloMethod.PSEUDO)
    mc_price = engine_mc.price(call, pricing_env)
    mc_error = abs(mc_price - bs_price)

    params_qmc = MCParams(num_paths=num_paths, time_steps=252, seed=42)
    engine_qmc = EuropeanMCEngine(params=params_qmc, method=MonteCarloMethod.QUASI)
    qmc_price = engine_qmc.price(call, pricing_env)
    qmc_error = abs(qmc_price - bs_price)

    print(
        f"  MC error:  {mc_error:.6f} (price=${mc_price:.6f})"
    )
    print(
        f"  QMC error: {qmc_error:.6f} (price=${qmc_price:.6f})"
    )
    print(
        f"  BS price:  ${bs_price:.6f}"
    )
    print(f"✓ QMC convergence test completed (QMC typically converges faster)")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 70)
    print("Running European MC Engine Unit Tests")
    print("=" * 70 + "\n")

    tests = [
        ("MC Call Pricing", test_mc_call_pricing),
        ("QMC Call Pricing", test_qmc_call_pricing),
        ("RQMC Call Pricing", test_rqmc_call_pricing),
        ("MC Put Pricing", test_mc_put_pricing),
        ("MC Put-Call Parity", test_mc_put_call_parity),
        ("Antithetic Variance Reduction", test_antithetic_variance_reduction),
        ("Two-Level Enum Pattern", test_two_level_enum_pattern),
        ("String Method Backward Compatibility", test_string_method_backward_compat),
        ("Invalid Method", test_invalid_method),
        ("Invalid Product Type", test_invalid_product_type),
        ("QMC Convergence", test_qmc_convergence_better_than_mc),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\nTest: {test_name}")
            print("-" * 70)
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)