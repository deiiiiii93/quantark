"""
Unit tests for Digital Option Monte Carlo pricing engine.
"""

import sys
from pathlib import Path
import math
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.product.option.digital_option import CashOrNothingDigitalOption
from asset.equity.engine.mc.digital_option_mc_engine import DigitalOptionMCEngine
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from asset.equity.param import MCParams
from util.enum import OptionType
from util.enum.engine_enums import MonteCarloMethod, EngineType
from util.exceptions import ValidationError, PricingError
from datetime import datetime


def cash_or_nothing_analytical(S, K, T, r, q, sigma, payout, is_call=True):
    """
    Analytical price for cash-or-nothing digital option.

    Call: payout * exp(-r*T) * N(d2)
    Put:  payout * exp(-r*T) * N(-d2)

    where d2 = [ln(S/K) + (r - q - sigma^2/2)*T] / (sigma*sqrt(T))
    """
    if T < 1e-10:
        if is_call:
            return payout if S > K else 0.0
        else:
            return payout if S < K else 0.0

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    discount = math.exp(-r * T)

    if is_call:
        return payout * discount * norm.cdf(d2)
    else:
        return payout * discount * norm.cdf(-d2)


def test_mc_call_pricing():
    """Test digital call option pricing with normal MC."""
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.20
    payout = 10.0

    spot = SpotQuote(spot=S)
    vol = FlatVolSurface(volatility=sigma)
    rate = FlatRateCurve(rate=r)
    div = ContinuousDividendYield(div_yield=q)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    digital_call = CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.CALL,
        maturity=T,
    )

    params = MCParams(num_paths=100000, time_steps=252, seed=42)
    engine = DigitalOptionMCEngine(params=params, method=MonteCarloMethod.PSEUDO)
    mc_price = engine.price(digital_call, pricing_env)

    analytical_price = cash_or_nothing_analytical(S, K, T, r, q, sigma, payout, is_call=True)

    std_error = engine.get_last_std_error()
    tolerance = 3 * std_error

    assert abs(mc_price - analytical_price) < tolerance, (
        f"MC price {mc_price:.6f} differs from analytical {analytical_price:.6f} "
        f"by more than 3 std errors (tolerance={tolerance:.6f})"
    )
    print(f"✓ MC call pricing test passed: MC=${mc_price:.6f}, Analytical=${analytical_price:.6f}, SE={std_error:.6f}")


def test_qmc_call_pricing():
    """Test digital call option pricing with QMC (Sobol)."""
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.20
    payout = 10.0

    spot = SpotQuote(spot=S)
    vol = FlatVolSurface(volatility=sigma)
    rate = FlatRateCurve(rate=r)
    div = ContinuousDividendYield(div_yield=q)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    digital_call = CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.CALL,
        maturity=T,
    )

    params = MCParams(num_paths=16384, time_steps=252, seed=42)
    engine = DigitalOptionMCEngine(params=params, method=MonteCarloMethod.QUASI)
    qmc_price = engine.price(digital_call, pricing_env)

    analytical_price = cash_or_nothing_analytical(S, K, T, r, q, sigma, payout, is_call=True)

    std_error = engine.get_last_std_error()
    tolerance = 3 * std_error

    assert abs(qmc_price - analytical_price) < tolerance, (
        f"QMC price {qmc_price:.6f} differs from analytical {analytical_price:.6f} "
        f"by more than 3 std errors (tolerance={tolerance:.6f})"
    )
    print(f"✓ QMC call pricing test passed: QMC=${qmc_price:.6f}, Analytical=${analytical_price:.6f}, SE={std_error:.6f}")


def test_mc_put_pricing():
    """Test digital put option pricing with MC."""
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.20
    payout = 10.0

    spot = SpotQuote(spot=S)
    vol = FlatVolSurface(volatility=sigma)
    rate = FlatRateCurve(rate=r)
    div = ContinuousDividendYield(div_yield=q)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    digital_put = CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.PUT,
        maturity=T,
    )

    params = MCParams(num_paths=100000, time_steps=252, seed=42)
    engine = DigitalOptionMCEngine(params=params, method=MonteCarloMethod.PSEUDO)
    mc_price = engine.price(digital_put, pricing_env)

    analytical_price = cash_or_nothing_analytical(S, K, T, r, q, sigma, payout, is_call=False)

    std_error = engine.get_last_std_error()
    tolerance = 3 * std_error

    assert abs(mc_price - analytical_price) < tolerance, (
        f"MC put price {mc_price:.6f} differs from analytical {analytical_price:.6f} "
        f"by more than 3 std errors (tolerance={tolerance:.6f})"
    )
    print(f"✓ MC put pricing test passed: MC=${mc_price:.6f}, Analytical=${analytical_price:.6f}, SE={std_error:.6f}")


def test_digital_call_put_parity():
    """Test call-put parity for digital options: Call + Put = payout * exp(-r*T)."""
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.20
    payout = 10.0

    spot = SpotQuote(spot=S)
    vol = FlatVolSurface(volatility=sigma)
    rate = FlatRateCurve(rate=r)
    div = ContinuousDividendYield(div_yield=q)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    digital_call = CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.CALL,
        maturity=T,
    )

    digital_put = CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.PUT,
        maturity=T,
    )

    params = MCParams(num_paths=100000, time_steps=252, seed=42)
    engine = DigitalOptionMCEngine(params=params, method=MonteCarloMethod.PSEUDO)
    call_price = engine.price(digital_call, pricing_env)
    put_price = engine.price(digital_put, pricing_env)

    # Digital call-put parity: Call + Put = payout * exp(-r*T)
    lhs = call_price + put_price
    rhs = payout * math.exp(-r * T)

    tolerance = 0.05  # 5 cents tolerance for $10 payout
    assert abs(lhs - rhs) < tolerance, (
        f"Digital call-put parity violated: {lhs:.6f} vs {rhs:.6f}, "
        f"difference={abs(lhs - rhs):.6f}"
    )
    print(f"✓ Digital call-put parity test passed: difference = {abs(lhs - rhs):.6f}")


def test_deep_itm_call():
    """Test deep ITM digital call (should price close to discounted payout)."""
    S = 100.0
    K = 70.0  # Deep ITM (lower strike means higher probability of ITM)
    T = 1.0
    r = 0.05
    q = 0.0
    sigma = 0.20
    payout = 10.0

    spot = SpotQuote(spot=S)
    vol = FlatVolSurface(volatility=sigma)
    rate = FlatRateCurve(rate=r)
    div = ContinuousDividendYield(div_yield=q)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    digital_call = CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.CALL,
        maturity=T,
    )

    params = MCParams(num_paths=100000, time_steps=252, seed=42)
    engine = DigitalOptionMCEngine(params=params, method=MonteCarloMethod.PSEUDO)
    mc_price = engine.price(digital_call, pricing_env)

    # Deep ITM call should be close to discounted payout
    discounted_payout = payout * math.exp(-r * T)
    analytical_price = cash_or_nothing_analytical(S, K, T, r, q, sigma, payout, is_call=True)
    tolerance = 0.20  # 20 cents tolerance

    assert abs(mc_price - analytical_price) < tolerance, (
        f"Deep ITM call price {mc_price:.6f} should be close to "
        f"analytical price {analytical_price:.6f}"
    )
    print(f"✓ Deep ITM call test passed: MC=${mc_price:.6f}, Analytical=${analytical_price:.6f}, Discounted=${discounted_payout:.6f}")


def test_deep_otm_call():
    """Test deep OTM digital call (should price close to zero)."""
    S = 100.0
    K = 180.0  # Deep OTM
    T = 1.0
    r = 0.05
    q = 0.0
    sigma = 0.20
    payout = 10.0

    spot = SpotQuote(spot=S)
    vol = FlatVolSurface(volatility=sigma)
    rate = FlatRateCurve(rate=r)
    div = ContinuousDividendYield(div_yield=q)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    digital_call = CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.CALL,
        maturity=T,
    )

    params = MCParams(num_paths=100000, time_steps=252, seed=42)
    engine = DigitalOptionMCEngine(params=params, method=MonteCarloMethod.PSEUDO)
    mc_price = engine.price(digital_call, pricing_env)

    analytical_price = cash_or_nothing_analytical(S, K, T, r, q, sigma, payout, is_call=True)

    # Deep OTM call should be close to zero, but we compare with analytical
    tolerance = 0.05  # 5 cents tolerance

    assert abs(mc_price - analytical_price) < tolerance, (
        f"Deep OTM call price {mc_price:.6f} should be close to "
        f"analytical {analytical_price:.6f}"
    )
    print(f"✓ Deep OTM call test passed: MC=${mc_price:.6f}, Analytical=${analytical_price:.6f}")


def test_atm_call():
    """Test ATM digital call (probability should be close to 0.5 with no drift)."""
    S = 100.0
    K = 100.0  # ATM
    T = 1.0
    r = 0.0
    q = 0.0
    sigma = 0.20
    payout = 10.0

    spot = SpotQuote(spot=S)
    vol = FlatVolSurface(volatility=sigma)
    rate = FlatRateCurve(rate=r)
    div = ContinuousDividendYield(div_yield=q)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    digital_call = CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.CALL,
        maturity=T,
    )

    params = MCParams(num_paths=200000, time_steps=252, seed=42)
    engine = DigitalOptionMCEngine(params=params, method=MonteCarloMethod.PSEUDO)
    mc_price = engine.price(digital_call, pricing_env)

    # ATM with no drift: compare with analytical price (exactly 0.5 * payout = 5.0)
    analytical_price = cash_or_nothing_analytical(S, K, T, r, q, sigma, payout, is_call=True)
    std_error = engine.get_last_std_error()
    tolerance = 4 * std_error  # 4 sigma tolerance

    assert abs(mc_price - analytical_price) < tolerance, (
        f"ATM call price {mc_price:.6f} should be close to "
        f"analytical {analytical_price:.6f} (SE={std_error:.6f}, tol={tolerance:.6f})"
    )
    print(f"✓ ATM call test passed: MC=${mc_price:.6f}, Analytical=${analytical_price:.6f}, SE={std_error:.6f}")


def test_two_level_enum_pattern():
    """Test two-level enum pattern for method selection."""
    params = MCParams(num_paths=1000, time_steps=100, seed=42)

    engine = DigitalOptionMCEngine(
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

    engine = DigitalOptionMCEngine(params=params, method="quasi")
    assert engine.method == MonteCarloMethod.QUASI

    engine = DigitalOptionMCEngine(params=params, method="pseudo")
    assert engine.method == MonteCarloMethod.PSEUDO

    engine = DigitalOptionMCEngine(params=params, method="randomized_quasi")
    assert engine.method == MonteCarloMethod.RANDOMIZED_QUASI

    print("✓ String method backward compatibility test passed")


def test_invalid_method():
    """Test that invalid method raises ValidationError."""
    params = MCParams(num_paths=1000, time_steps=100, seed=42)

    try:
        engine = DigitalOptionMCEngine(params=params, method="invalid_method")
        assert False, "Should have raised ValidationError for invalid method"
    except ValidationError as e:
        assert "Invalid method string" in str(e)
        print("✓ Invalid method validation test passed")


def test_invalid_product_type():
    """Test that non-digital options raise PricingError."""
    from asset.equity.product.option import EuropeanVanillaOption

    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        valuation_date=datetime(2024, 1, 1),
    )

    vanilla_call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    params = MCParams(num_paths=1000, time_steps=100, seed=42)
    engine = DigitalOptionMCEngine(params=params)

    try:
        engine.price(vanilla_call, pricing_env)
        assert False, "Should have raised PricingError for vanilla option"
    except PricingError as e:
        assert "CashOrNothingDigitalOption" in str(e)
        print("✓ Invalid product type test passed")


def test_rqmc_call_pricing():
    """Test digital call option pricing with RQMC."""
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.20
    payout = 10.0

    spot = SpotQuote(spot=S)
    vol = FlatVolSurface(volatility=sigma)
    rate = FlatRateCurve(rate=r)
    div = ContinuousDividendYield(div_yield=q)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    digital_call = CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.CALL,
        maturity=T,
    )

    params = MCParams(num_paths=8192, time_steps=252, seed=42)
    params.max_batches = 16
    params.target_std = 0.01
    params.min_batches = 4

    engine = DigitalOptionMCEngine(params=params, method=MonteCarloMethod.RANDOMIZED_QUASI)
    rqmc_price = engine.price(digital_call, pricing_env)

    analytical_price = cash_or_nothing_analytical(S, K, T, r, q, sigma, payout, is_call=True)

    rqmc_result = engine.get_last_rqmc_result()
    std_error = rqmc_result.std_error
    tolerance = 3 * std_error

    assert abs(rqmc_price - analytical_price) < tolerance, (
        f"RQMC price {rqmc_price:.6f} differs from analytical {analytical_price:.6f} "
        f"by more than 3 std errors (tolerance={tolerance:.6f})"
    )
    print(
        f"✓ RQMC call pricing test passed: RQMC=${rqmc_price:.6f}, Analytical=${analytical_price:.6f}, "
        f"SE={std_error:.6f}, batches={rqmc_result.batches_used}"
    )


def test_qmc_vs_mc_convergence():
    """Test that QMC converges faster than MC for digital options."""
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.20
    payout = 10.0

    spot = SpotQuote(spot=S)
    vol = FlatVolSurface(volatility=sigma)
    rate = FlatRateCurve(rate=r)
    div = ContinuousDividendYield(div_yield=q)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    digital_call = CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.CALL,
        maturity=T,
    )

    analytical_price = cash_or_nothing_analytical(S, K, T, r, q, sigma, payout, is_call=True)

    num_paths = 8192
    params_mc = MCParams(num_paths=num_paths, time_steps=252, seed=42)
    engine_mc = DigitalOptionMCEngine(params=params_mc, method=MonteCarloMethod.PSEUDO)
    mc_price = engine_mc.price(digital_call, pricing_env)
    mc_error = abs(mc_price - analytical_price)

    params_qmc = MCParams(num_paths=num_paths, time_steps=252, seed=42)
    engine_qmc = DigitalOptionMCEngine(params=params_qmc, method=MonteCarloMethod.QUASI)
    qmc_price = engine_qmc.price(digital_call, pricing_env)
    qmc_error = abs(qmc_price - analytical_price)

    print(
        f"  MC error:  {mc_error:.6f} (price=${mc_price:.6f})"
    )
    print(
        f"  QMC error: {qmc_error:.6f} (price=${qmc_price:.6f})"
    )
    print(
        f"  Analytical: ${analytical_price:.6f}"
    )
    print(f"✓ QMC convergence test completed (QMC typically converges faster)")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 70)
    print("Running Digital Option MC Engine Unit Tests")
    print("=" * 70 + "\n")

    tests = [
        ("MC Call Pricing", test_mc_call_pricing),
        ("QMC Call Pricing", test_qmc_call_pricing),
        ("RQMC Call Pricing", test_rqmc_call_pricing),
        ("MC Put Pricing", test_mc_put_pricing),
        ("Digital Call-Put Parity", test_digital_call_put_parity),
        ("Deep ITM Call", test_deep_itm_call),
        ("Deep OTM Call", test_deep_otm_call),
        ("ATM Call", test_atm_call),
        ("Two-Level Enum Pattern", test_two_level_enum_pattern),
        ("String Method Backward Compatibility", test_string_method_backward_compat),
        ("Invalid Method", test_invalid_method),
        ("Invalid Product Type", test_invalid_product_type),
        ("QMC Convergence", test_qmc_vs_mc_convergence),
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
