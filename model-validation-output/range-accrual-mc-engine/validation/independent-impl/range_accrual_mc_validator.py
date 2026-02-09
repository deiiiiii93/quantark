"""
Independent validation implementation for Range Accrual Monte Carlo pricing.

This is Developer B's independent implementation for cross-validation of the
Range Accrual MC Engine. The implementation prioritizes clarity and correctness
over performance.

Author: Developer B (Validation)
Purpose: Independent verification of Range Accrual MC pricing logic
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result from validation pricing."""
    price: float
    std_error: float
    num_paths: int
    seed: int


def price_range_accrual_mc(
    # Product parameters
    initial_price: float,
    upper_barrier: float,
    lower_barrier: float,
    accrual_rate: float,
    maturity: float,
    contract_multiplier: float,
    observation_times: List[float],
    observation_weights: Optional[List[float]] = None,
    is_rate_annualized: bool = False,
    is_reverse: bool = False,
    # Market parameters
    spot: float = 100.0,
    volatility: float = 0.2,
    risk_free_rate: float = 0.05,
    div_yield: float = 0.0,
    # Simulation parameters
    num_paths: int = 100000,
    seed: int = 42,
) -> ValidationResult:
    """
    Price a Range Accrual option using Monte Carlo simulation.

    This is a simple, clear implementation for validation purposes.
    Uses standard GBM simulation without variance reduction.

    Payoff Formula:
        Payoff = initial_price * contract_multiplier * accrual_rate
                 * (sum_in_range_weights / sum_total_weights) * year_fraction

    Args:
        initial_price: Reference/initial price for payoff calculation
        upper_barrier: Upper barrier level
        lower_barrier: Lower barrier level
        accrual_rate: Accrual rate (per-period or annualized)
        maturity: Time to maturity in years
        contract_multiplier: Contract multiplier
        observation_times: List of observation times as year fractions
        observation_weights: Optional weights for each observation (default: all 1.0)
        is_rate_annualized: If True, multiply payoff by year_fraction
        is_reverse: If True, accrue when OUTSIDE range
        spot: Current spot price
        volatility: Annualized volatility
        risk_free_rate: Risk-free rate
        div_yield: Dividend yield
        num_paths: Number of Monte Carlo paths
        seed: Random seed for reproducibility

    Returns:
        ValidationResult with price, std_error, and simulation info
    """
    np.random.seed(seed)

    # Default weights to 1.0 for each observation
    if observation_weights is None:
        observation_weights = [1.0] * len(observation_times)

    # Sort observations by time
    obs_sorted = sorted(zip(observation_times, observation_weights))
    obs_times = [t for t, w in obs_sorted]
    obs_weights = [w for t, w in obs_sorted]

    # Total weight for normalization
    total_weight = sum(obs_weights)

    # Year fraction for payoff
    year_fraction = maturity if is_rate_annualized else 1.0

    # Pre-compute drift and vol for GBM
    drift = risk_free_rate - div_yield

    # Simulate paths and compute payoffs
    payoffs = np.zeros(num_paths)

    for path_idx in range(num_paths):
        # Simulate spot at each observation time
        current_spot = spot
        current_time = 0.0
        in_range_weight = 0.0

        for obs_idx, (obs_time, obs_weight) in enumerate(zip(obs_times, obs_weights)):
            dt = obs_time - current_time

            if dt > 0:
                # GBM step: S(t+dt) = S(t) * exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
                z = np.random.standard_normal()
                current_spot = current_spot * np.exp(
                    (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z
                )

            current_time = obs_time

            # Check if spot is in range
            in_range = lower_barrier <= current_spot <= upper_barrier

            # Reverse mode: accrue when OUTSIDE range
            if is_reverse:
                in_range = not in_range

            if in_range:
                in_range_weight += obs_weight

        # Calculate accrual ratio
        if total_weight > 0:
            accrual_ratio = in_range_weight / total_weight
        else:
            accrual_ratio = 0.0

        # Calculate payoff
        payoff = (
            initial_price
            * contract_multiplier
            * accrual_rate
            * accrual_ratio
            * year_fraction
        )

        payoffs[path_idx] = payoff

    # Discount to present value
    discount_factor = np.exp(-risk_free_rate * maturity)
    discounted_payoffs = payoffs * discount_factor

    # Calculate price and standard error
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(num_paths)

    return ValidationResult(
        price=price,
        std_error=std_error,
        num_paths=num_paths,
        seed=seed,
    )


def price_range_accrual_from_product(
    option,  # RangeAccrualOption
    pricing_env,  # PricingEnvironment
    num_paths: int = 100000,
    seed: int = 42,
) -> ValidationResult:
    """
    Price a RangeAccrualOption using the validation Monte Carlo implementation.

    This function extracts parameters from the product and pricing environment
    and calls the core pricing function.

    Args:
        option: RangeAccrualOption instance
        pricing_env: PricingEnvironment instance
        num_paths: Number of Monte Carlo paths
        seed: Random seed

    Returns:
        ValidationResult with price and statistics
    """
    # Extract product parameters
    initial_price = option.initial_price
    config = option.range_config

    # Get barriers (handle scalar vs list)
    if isinstance(config.upper_barrier, list):
        # For time-varying barriers, we need per-observation checking
        raise NotImplementedError(
            "Time-varying barriers not yet supported in validation impl. "
            "Use scalar barriers for validation."
        )
    upper_barrier = config.upper_barrier
    lower_barrier = config.lower_barrier

    accrual_rate = config.accrual_rate
    is_rate_annualized = config.is_rate_annualized
    is_reverse = config.is_reverse

    maturity = option.maturity
    contract_multiplier = option.contract_multiplier

    # Get observation schedule
    records = option.get_observation_records()
    observation_times = []
    observation_weights = []

    for rec in records:
        # Use observation_time directly (assume all future for simplicity)
        t = rec.observation_time if rec.observation_time is not None else 0.0
        observation_times.append(t)
        observation_weights.append(rec.weight)

    # Extract market parameters
    spot = pricing_env.spot
    volatility = pricing_env.get_vol(initial_price, maturity)
    risk_free_rate = pricing_env.get_rate(maturity)
    div_yield = pricing_env.get_div_yield(maturity)

    return price_range_accrual_mc(
        initial_price=initial_price,
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
        accrual_rate=accrual_rate,
        maturity=maturity,
        contract_multiplier=contract_multiplier,
        observation_times=observation_times,
        observation_weights=observation_weights,
        is_rate_annualized=is_rate_annualized,
        is_reverse=is_reverse,
        spot=spot,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        div_yield=div_yield,
        num_paths=num_paths,
        seed=seed,
    )


def run_validation_tests():
    """
    Run validation tests comparing independent implementation with Developer A's engine.
    """
    import sys
    sys.path.insert(0, '/Users/fuxinyao/quant-ark')

    from datetime import datetime
    from asset.equity.product.option import RangeAccrualOption
    from asset.equity.product.option.range_accrual_config import RangeAccrualConfig
    from asset.equity.product.option.range_accrual_helpers import create_standard_range_accrual
    from asset.equity.engine.mc.range_accrual_mc_engine import RangeAccrualMCEngine
    from asset.equity.param import MCParams
    from priceenv import PricingEnvironment
    from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield

    print("=" * 70)
    print("RANGE ACCRUAL MC ENGINE - INDEPENDENT VALIDATION")
    print("=" * 70)
    print()

    results = []

    # Test Case 1: Basic Range Accrual
    print("Test Case 1: Basic Range Accrual")
    print("-" * 50)

    option1 = create_standard_range_accrual(
        initial_price=100.0,
        upper_barrier=110.0,
        lower_barrier=90.0,
        maturity=1.0,
        accrual_rate=0.05,
        num_observations=12,  # Monthly
        is_rate_annualized=True,
    )

    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2025, 1, 1),
    )

    # Price with validation implementation
    num_paths = 100000
    seed = 42

    val_result = price_range_accrual_from_product(
        option1, pricing_env, num_paths=num_paths, seed=seed
    )

    # Price with Developer A's engine
    engine = RangeAccrualMCEngine(
        params=MCParams(num_paths=num_paths, seed=seed)
    )
    dev_a_price = engine.price(option1, pricing_env)

    diff = abs(val_result.price - dev_a_price)
    within_tolerance = diff < 3 * val_result.std_error  # 3 sigma tolerance

    print(f"  Initial Price: {option1.initial_price}")
    print(f"  Barriers: [{option1.range_config.lower_barrier}, {option1.range_config.upper_barrier}]")
    print(f"  Accrual Rate: {option1.range_config.accrual_rate:.2%}")
    print(f"  Maturity: {option1.maturity} years")
    print(f"  Observations: {len(option1.get_observation_records())}")
    print()
    print(f"  Validation Price:   {val_result.price:.6f} (+/- {val_result.std_error:.6f})")
    print(f"  Developer A Price:  {dev_a_price:.6f}")
    print(f"  Difference:         {diff:.6f}")
    print(f"  Within 3-sigma:     {'PASS' if within_tolerance else 'FAIL'}")
    print()

    results.append({
        'test': 'Basic Range Accrual',
        'val_price': val_result.price,
        'dev_a_price': dev_a_price,
        'std_error': val_result.std_error,
        'diff': diff,
        'pass': within_tolerance,
    })

    # Test Case 2: Low volatility (most observations in range)
    print("Test Case 2: Low Volatility (High In-Range Probability)")
    print("-" * 50)

    pricing_env_low_vol = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.10),  # Low vol
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2025, 1, 1),
    )

    val_result2 = price_range_accrual_from_product(
        option1, pricing_env_low_vol, num_paths=num_paths, seed=seed
    )

    engine2 = RangeAccrualMCEngine(
        params=MCParams(num_paths=num_paths, seed=seed)
    )
    dev_a_price2 = engine2.price(option1, pricing_env_low_vol)

    diff2 = abs(val_result2.price - dev_a_price2)
    within_tolerance2 = diff2 < 3 * val_result2.std_error

    print(f"  Volatility: 10%")
    print(f"  Validation Price:   {val_result2.price:.6f} (+/- {val_result2.std_error:.6f})")
    print(f"  Developer A Price:  {dev_a_price2:.6f}")
    print(f"  Difference:         {diff2:.6f}")
    print(f"  Within 3-sigma:     {'PASS' if within_tolerance2 else 'FAIL'}")
    print()

    results.append({
        'test': 'Low Volatility',
        'val_price': val_result2.price,
        'dev_a_price': dev_a_price2,
        'std_error': val_result2.std_error,
        'diff': diff2,
        'pass': within_tolerance2,
    })

    # Test Case 3: High volatility (fewer observations in range)
    print("Test Case 3: High Volatility (Low In-Range Probability)")
    print("-" * 50)

    pricing_env_high_vol = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.40),  # High vol
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2025, 1, 1),
    )

    val_result3 = price_range_accrual_from_product(
        option1, pricing_env_high_vol, num_paths=num_paths, seed=seed
    )

    engine3 = RangeAccrualMCEngine(
        params=MCParams(num_paths=num_paths, seed=seed)
    )
    dev_a_price3 = engine3.price(option1, pricing_env_high_vol)

    diff3 = abs(val_result3.price - dev_a_price3)
    within_tolerance3 = diff3 < 3 * val_result3.std_error

    print(f"  Volatility: 40%")
    print(f"  Validation Price:   {val_result3.price:.6f} (+/- {val_result3.std_error:.6f})")
    print(f"  Developer A Price:  {dev_a_price3:.6f}")
    print(f"  Difference:         {diff3:.6f}")
    print(f"  Within 3-sigma:     {'PASS' if within_tolerance3 else 'FAIL'}")
    print()

    results.append({
        'test': 'High Volatility',
        'val_price': val_result3.price,
        'dev_a_price': dev_a_price3,
        'std_error': val_result3.std_error,
        'diff': diff3,
        'pass': within_tolerance3,
    })

    # Test Case 4: Narrow range
    print("Test Case 4: Narrow Range (95-105)")
    print("-" * 50)

    option_narrow = create_standard_range_accrual(
        initial_price=100.0,
        upper_barrier=105.0,  # Narrow range
        lower_barrier=95.0,
        maturity=1.0,
        accrual_rate=0.08,  # Higher rate for narrow range
        num_observations=12,
        is_rate_annualized=True,
    )

    val_result4 = price_range_accrual_from_product(
        option_narrow, pricing_env, num_paths=num_paths, seed=seed
    )

    engine4 = RangeAccrualMCEngine(
        params=MCParams(num_paths=num_paths, seed=seed)
    )
    dev_a_price4 = engine4.price(option_narrow, pricing_env)

    diff4 = abs(val_result4.price - dev_a_price4)
    within_tolerance4 = diff4 < 3 * val_result4.std_error

    print(f"  Barriers: [95, 105]")
    print(f"  Validation Price:   {val_result4.price:.6f} (+/- {val_result4.std_error:.6f})")
    print(f"  Developer A Price:  {dev_a_price4:.6f}")
    print(f"  Difference:         {diff4:.6f}")
    print(f"  Within 3-sigma:     {'PASS' if within_tolerance4 else 'FAIL'}")
    print()

    results.append({
        'test': 'Narrow Range',
        'val_price': val_result4.price,
        'dev_a_price': dev_a_price4,
        'std_error': val_result4.std_error,
        'diff': diff4,
        'pass': within_tolerance4,
    })

    # Test Case 5: Many observations
    print("Test Case 5: Daily Observations (252)")
    print("-" * 50)

    option_daily = create_standard_range_accrual(
        initial_price=100.0,
        upper_barrier=110.0,
        lower_barrier=90.0,
        maturity=1.0,
        accrual_rate=0.05,
        num_observations=252,  # Daily
        is_rate_annualized=True,
    )

    val_result5 = price_range_accrual_from_product(
        option_daily, pricing_env, num_paths=50000, seed=seed  # Fewer paths for speed
    )

    engine5 = RangeAccrualMCEngine(
        params=MCParams(num_paths=50000, seed=seed)
    )
    dev_a_price5 = engine5.price(option_daily, pricing_env)

    diff5 = abs(val_result5.price - dev_a_price5)
    within_tolerance5 = diff5 < 3 * val_result5.std_error

    print(f"  Observations: 252 (daily)")
    print(f"  Validation Price:   {val_result5.price:.6f} (+/- {val_result5.std_error:.6f})")
    print(f"  Developer A Price:  {dev_a_price5:.6f}")
    print(f"  Difference:         {diff5:.6f}")
    print(f"  Within 3-sigma:     {'PASS' if within_tolerance5 else 'FAIL'}")
    print()

    results.append({
        'test': 'Daily Observations',
        'val_price': val_result5.price,
        'dev_a_price': dev_a_price5,
        'std_error': val_result5.std_error,
        'diff': diff5,
        'pass': within_tolerance5,
    })

    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_pass = all(r['pass'] for r in results)

    for r in results:
        status = "PASS" if r['pass'] else "FAIL"
        print(f"  {r['test']}: {status} (diff={r['diff']:.6f}, 3*se={3*r['std_error']:.6f})")

    print()
    print(f"Overall Result: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print()

    return results, all_pass


if __name__ == '__main__':
    results, all_pass = run_validation_tests()
