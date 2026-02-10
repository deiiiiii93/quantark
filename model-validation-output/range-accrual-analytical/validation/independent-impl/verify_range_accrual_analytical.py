"""
Independent verification of Range Accrual Analytical Engine (Developer B).

SR 11-7 Model Validation: This script independently implements the Range Accrual
pricing formula from the research specification (digital decomposition approach)
and compares results against Developer A's implementation.

The formula is derived independently from:
    model-validation-output/range-accrual-analytical/research/research-report.md

Core Formula:
    Price = exp(-r*T) * S_0 * M * c * tau * (1/W) * [past_in_range + sum_i w_i * P_i]

    P_i = N(d2_L) - N(d2_U)   (standard mode)
    P_i = 1 - [N(d2_L) - N(d2_U)]  (reverse mode)

    d2(K, t_i) = [ln(S/K) + (r - q - sigma^2/2) * t_i] / (sigma * sqrt(t_i))

Author: Developer B (Independent Verification)
"""

import math
import sys
import os
from datetime import datetime
from typing import List, Optional, Tuple

from scipy.stats import norm


# =============================================================================
# INDEPENDENT IMPLEMENTATION (from research report only)
# =============================================================================

def d2(S: float, K: float, r: float, q: float, sigma: float, t: float) -> float:
    """
    Compute d2 for digital option probability under risk-neutral measure.

    d2(K, t) = [ln(S/K) + (r - q - sigma^2/2) * t] / (sigma * sqrt(t))

    Args:
        S: Current spot price
        K: Barrier level
        r: Risk-free rate (continuous)
        q: Dividend yield (continuous)
        sigma: Volatility
        t: Time to observation (years)

    Returns:
        d2 value
    """
    if t <= 1e-10:
        # Near-expiry: deterministic
        # Return +inf if S > K (N(d2) -> 1), -inf if S < K (N(d2) -> 0)
        if S > K:
            return float('inf')
        elif S < K:
            return float('-inf')
        else:
            return 0.0

    if sigma < 1e-8:
        # Near-zero vol: use forward price deterministic check
        fwd = S * math.exp((r - q) * t)
        if fwd > K:
            return float('inf')
        elif fwd < K:
            return float('-inf')
        else:
            return 0.0

    numerator = math.log(S / K) + (r - q - 0.5 * sigma * sigma) * t
    denominator = sigma * math.sqrt(t)
    return numerator / denominator


def in_range_probability(
    S: float, L: float, U: float, r: float, q: float, sigma: float, t: float
) -> float:
    """
    Compute probability that S(t) is in [L, U] under risk-neutral measure.

    P = N(d2_L) - N(d2_U)

    where d2_L = d2(S, L, ...) and d2_U = d2(S, U, ...).

    N(d2_L) = P(S(t) >= L) and N(d2_U) = P(S(t) >= U),
    so P(L <= S(t) <= U) = N(d2_L) - N(d2_U).

    Args:
        S: Current spot price
        L: Lower barrier
        U: Upper barrier
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        t: Time to observation

    Returns:
        Probability that S(t) is in [L, U]
    """
    if t <= 1e-10:
        # Deterministic: check if S is in range
        return 1.0 if L <= S <= U else 0.0

    if sigma < 1e-8:
        # Zero vol: forward is deterministic
        fwd = S * math.exp((r - q) * t)
        return 1.0 if L <= fwd <= U else 0.0

    d2_lower = d2(S, L, r, q, sigma, t)
    d2_upper = d2(S, U, r, q, sigma, t)

    prob = norm.cdf(d2_lower) - norm.cdf(d2_upper)
    return max(0.0, prob)


def price_range_accrual(
    S: float,
    lower_barriers: List[float],
    upper_barriers: List[float],
    observation_times: List[float],
    observation_weights: List[float],
    r: float,
    q: float,
    sigma: float,
    T: float,
    accrual_rate: float,
    contract_multiplier: float,
    is_rate_annualized: bool,
    is_reverse: bool = False,
    past_observations: Optional[List[Tuple[float, bool]]] = None,
) -> Tuple[float, float, List[float]]:
    """
    Price a Range Accrual option using digital decomposition.

    Formula:
        Price = exp(-r*T) * S * M * c * tau * (1/W) * [past_accrual + sum_i w_i * P_i]

    Where:
        - exp(-r*T): discount factor to maturity
        - S: initial/reference price (= S_0)
        - M: contract multiplier
        - c: accrual rate
        - tau: year fraction (T if annualized, 1.0 otherwise)
        - W: total weights (sum of all observation weights)
        - past_accrual: sum of weights for past in-range observations
        - P_i: probability observation i is in range

    Args:
        S: Spot/initial price
        lower_barriers: Lower barrier for each future observation
        upper_barriers: Upper barrier for each future observation
        observation_times: Time to each future observation (years)
        observation_weights: Weight of each future observation
        r: Risk-free rate
        q: Dividend yield
        sigma: Volatility
        T: Time to maturity
        accrual_rate: Accrual rate (per period or annualized)
        contract_multiplier: Contract multiplier
        is_rate_annualized: Whether rate is annualized
        is_reverse: Whether reverse mode (pay outside range)
        past_observations: List of (weight, in_range_bool) for past obs

    Returns:
        Tuple of (price, expected_ratio, per_obs_probabilities)
    """
    # Year fraction: T if annualized, 1.0 otherwise
    tau = T if is_rate_annualized else 1.0

    # Compute total weights (past + future)
    total_weights = sum(observation_weights)
    if past_observations is not None:
        total_weights += sum(w for w, _ in past_observations)

    if total_weights <= 0:
        return 0.0, 0.0, []

    # Past accrual contribution
    past_in_range_weights = 0.0
    if past_observations is not None:
        for w, in_range in past_observations:
            if in_range:
                past_in_range_weights += w

    # Future: compute per-observation probabilities
    per_obs_probs = []
    expected_in_range_weights = past_in_range_weights

    for i in range(len(observation_times)):
        t_i = observation_times[i]
        L_i = lower_barriers[i]
        U_i = upper_barriers[i]
        w_i = observation_weights[i]

        prob_i = in_range_probability(S, L_i, U_i, r, q, sigma, t_i)

        if is_reverse:
            prob_i = 1.0 - prob_i

        per_obs_probs.append(prob_i)
        expected_in_range_weights += w_i * prob_i

    # Expected accrual ratio
    expected_ratio = expected_in_range_weights / total_weights

    # Discount factor
    df = math.exp(-r * T)

    # Price = df * S * M * c * tau * expected_ratio
    price = df * S * contract_multiplier * accrual_rate * tau * expected_ratio

    return price, expected_ratio, per_obs_probs


# =============================================================================
# TEST CASES
# =============================================================================

def run_test_case_1():
    """
    Case 1: Standard case.
    S=100, L=90, U=110, r=0.05, q=0.02, sigma=0.2, T=1.0,
    12 monthly obs, rate=0.05 annualized, mult=1.0
    """
    S = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.2
    T = 1.0
    n_obs = 12

    obs_times = [(i + 1) / 12 for i in range(n_obs)]
    obs_weights = [1.0] * n_obs
    lower_barriers = [90.0] * n_obs
    upper_barriers = [110.0] * n_obs

    price, ratio, probs = price_range_accrual(
        S=S,
        lower_barriers=lower_barriers,
        upper_barriers=upper_barriers,
        observation_times=obs_times,
        observation_weights=obs_weights,
        r=r, q=q, sigma=sigma, T=T,
        accrual_rate=0.05,
        contract_multiplier=1.0,
        is_rate_annualized=True,
        is_reverse=False,
    )

    return price, ratio, probs


def run_test_case_2():
    """
    Case 2: Reverse mode - same params as Case 1.
    Should give 1 - standard ratio for each observation.
    """
    S = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.2
    T = 1.0
    n_obs = 12

    obs_times = [(i + 1) / 12 for i in range(n_obs)]
    obs_weights = [1.0] * n_obs
    lower_barriers = [90.0] * n_obs
    upper_barriers = [110.0] * n_obs

    price, ratio, probs = price_range_accrual(
        S=S,
        lower_barriers=lower_barriers,
        upper_barriers=upper_barriers,
        observation_times=obs_times,
        observation_weights=obs_weights,
        r=r, q=q, sigma=sigma, T=T,
        accrual_rate=0.05,
        contract_multiplier=1.0,
        is_rate_annualized=True,
        is_reverse=True,
    )

    return price, ratio, probs


def run_test_case_3():
    """
    Case 3: Narrow range, low vol, shorter tenor.
    S=100, L=95, U=105, r=0.05, q=0.02, sigma=0.1, T=0.5,
    4 quarterly obs, rate=0.08 non-annualized
    """
    S = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.1
    T = 0.5
    n_obs = 4

    # 4 observations evenly spaced over 0.5 years
    obs_times = [(i + 1) / n_obs * T for i in range(n_obs)]
    obs_weights = [1.0] * n_obs
    lower_barriers = [95.0] * n_obs
    upper_barriers = [105.0] * n_obs

    price, ratio, probs = price_range_accrual(
        S=S,
        lower_barriers=lower_barriers,
        upper_barriers=upper_barriers,
        observation_times=obs_times,
        observation_weights=obs_weights,
        r=r, q=q, sigma=sigma, T=T,
        accrual_rate=0.08,
        contract_multiplier=1.0,
        is_rate_annualized=False,
        is_reverse=False,
    )

    return price, ratio, probs


def run_test_case_4():
    """
    Case 4: Time-varying barriers.
    L=[85,88,90,92], U=[115,112,110,108], S=100, r=0.05, q=0.02, sigma=0.2, T=1.0,
    4 observations at 0.25, 0.5, 0.75, 1.0, rate=0.05 annualized
    """
    S = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.2
    T = 1.0

    lower_barriers = [85.0, 88.0, 90.0, 92.0]
    upper_barriers = [115.0, 112.0, 110.0, 108.0]
    obs_times = [0.25, 0.5, 0.75, 1.0]
    obs_weights = [1.0] * 4

    price, ratio, probs = price_range_accrual(
        S=S,
        lower_barriers=lower_barriers,
        upper_barriers=upper_barriers,
        observation_times=obs_times,
        observation_weights=obs_weights,
        r=r, q=q, sigma=sigma, T=T,
        accrual_rate=0.05,
        contract_multiplier=1.0,
        is_rate_annualized=True,
        is_reverse=False,
    )

    return price, ratio, probs


def run_test_case_5():
    """
    Case 5: Past + future observations.
    2 past (one in-range, one out) + 2 future, rate=0.05 annualized.
    S=100, L=90, U=110, r=0.05, q=0.02, sigma=0.2, T=1.0
    Past obs at t=-0.25 (in_range=True), t=-0.083 (in_range=False)
    Future obs at t=0.25, t=0.5
    Maturity T=1.0 (from original inception, but remaining = we need to figure this out)
    """
    S = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.2
    # Maturity from now = 0.5 (the last observation is at t=0.5)
    # But T for discounting is the time to maturity from valuation date
    # Based on the product definition, maturity is from valuation date
    T = 1.0  # total maturity

    # Past observations: already occurred, have known outcomes
    past_observations = [
        (1.0, True),   # weight=1, was in range
        (1.0, False),  # weight=1, was NOT in range
    ]

    # Future observations
    future_obs_times = [0.25, 0.5]
    future_obs_weights = [1.0, 1.0]
    future_lower = [90.0, 90.0]
    future_upper = [110.0, 110.0]

    price, ratio, probs = price_range_accrual(
        S=S,
        lower_barriers=future_lower,
        upper_barriers=future_upper,
        observation_times=future_obs_times,
        observation_weights=future_obs_weights,
        r=r, q=q, sigma=sigma, T=T,
        accrual_rate=0.05,
        contract_multiplier=1.0,
        is_rate_annualized=True,
        is_reverse=False,
        past_observations=past_observations,
    )

    return price, ratio, probs


# =============================================================================
# DEVELOPER A ENGINE COMPARISON
# =============================================================================

def build_pricing_env(spot, vol, rate, div_yield):
    """Build a PricingEnvironment with flat market data."""
    from priceenv import PricingEnvironment
    from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield

    return PricingEnvironment(
        valuation_date=datetime(2024, 1, 1),
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
    )


def dev_a_case_1():
    """Run Developer A's engine for Case 1."""
    from asset.equity.engine.analytical import RangeAccrualAnalyticalEngine
    from asset.equity.product.option import RangeAccrualOption, RangeAccrualConfig

    env = build_pricing_env(100.0, 0.2, 0.05, 0.02)
    config = RangeAccrualConfig(
        upper_barrier=110.0,
        lower_barrier=90.0,
        accrual_rate=0.05,
        is_rate_annualized=True,
    )
    obs_times = [(i + 1) / 12 for i in range(12)]
    option = RangeAccrualOption(
        initial_price=100.0,
        range_config=config,
        observation_times=obs_times,
        maturity=1.0,
        contract_multiplier=1.0,
    )
    engine = RangeAccrualAnalyticalEngine()
    price = engine.price(option, env)
    result = engine.get_last_result()
    return price, result.expected_accrual_ratio, result.per_observation_probs


def dev_a_case_2():
    """Run Developer A's engine for Case 2 (reverse mode)."""
    from asset.equity.engine.analytical import RangeAccrualAnalyticalEngine
    from asset.equity.product.option import RangeAccrualOption, RangeAccrualConfig

    env = build_pricing_env(100.0, 0.2, 0.05, 0.02)
    config = RangeAccrualConfig(
        upper_barrier=110.0,
        lower_barrier=90.0,
        accrual_rate=0.05,
        is_rate_annualized=True,
        is_reverse=True,
    )
    obs_times = [(i + 1) / 12 for i in range(12)]
    option = RangeAccrualOption(
        initial_price=100.0,
        range_config=config,
        observation_times=obs_times,
        maturity=1.0,
        contract_multiplier=1.0,
    )
    engine = RangeAccrualAnalyticalEngine()
    price = engine.price(option, env)
    result = engine.get_last_result()
    return price, result.expected_accrual_ratio, result.per_observation_probs


def dev_a_case_3():
    """Run Developer A's engine for Case 3."""
    from asset.equity.engine.analytical import RangeAccrualAnalyticalEngine
    from asset.equity.product.option import RangeAccrualOption, RangeAccrualConfig

    env = build_pricing_env(100.0, 0.1, 0.05, 0.02)
    config = RangeAccrualConfig(
        upper_barrier=105.0,
        lower_barrier=95.0,
        accrual_rate=0.08,
        is_rate_annualized=False,
    )
    n_obs = 4
    T = 0.5
    obs_times = [(i + 1) / n_obs * T for i in range(n_obs)]
    option = RangeAccrualOption(
        initial_price=100.0,
        range_config=config,
        observation_times=obs_times,
        maturity=T,
        contract_multiplier=1.0,
    )
    engine = RangeAccrualAnalyticalEngine()
    price = engine.price(option, env)
    result = engine.get_last_result()
    return price, result.expected_accrual_ratio, result.per_observation_probs


def dev_a_case_4():
    """Run Developer A's engine for Case 4 (time-varying barriers)."""
    from asset.equity.engine.analytical import RangeAccrualAnalyticalEngine
    from asset.equity.product.option import RangeAccrualOption, RangeAccrualConfig

    env = build_pricing_env(100.0, 0.2, 0.05, 0.02)
    config = RangeAccrualConfig(
        upper_barrier=[115.0, 112.0, 110.0, 108.0],
        lower_barrier=[85.0, 88.0, 90.0, 92.0],
        accrual_rate=0.05,
        is_rate_annualized=True,
    )
    obs_times = [0.25, 0.5, 0.75, 1.0]
    option = RangeAccrualOption(
        initial_price=100.0,
        range_config=config,
        observation_times=obs_times,
        maturity=1.0,
        contract_multiplier=1.0,
    )
    engine = RangeAccrualAnalyticalEngine()
    price = engine.price(option, env)
    result = engine.get_last_result()
    return price, result.expected_accrual_ratio, result.per_observation_probs


def dev_a_case_5():
    """Run Developer A's engine for Case 5 (past + future obs)."""
    from asset.equity.engine.analytical import RangeAccrualAnalyticalEngine
    from asset.equity.product.option import (
        RangeAccrualOption,
        RangeAccrualConfig,
        RangeAccrualObservationRecord,
    )

    env = build_pricing_env(100.0, 0.2, 0.05, 0.02)
    config = RangeAccrualConfig(
        upper_barrier=110.0,
        lower_barrier=90.0,
        accrual_rate=0.05,
        is_rate_annualized=True,
    )
    records = [
        RangeAccrualObservationRecord(
            observation_time=-0.5, weight=1.0, observed_in_range=True,
        ),
        RangeAccrualObservationRecord(
            observation_time=-0.25, weight=1.0, observed_in_range=False,
        ),
        RangeAccrualObservationRecord(
            observation_time=0.25, weight=1.0,
        ),
        RangeAccrualObservationRecord(
            observation_time=0.5, weight=1.0,
        ),
    ]
    option = RangeAccrualOption(
        initial_price=100.0,
        range_config=config,
        observation_records=records,
        maturity=1.0,
        contract_multiplier=1.0,
    )
    engine = RangeAccrualAnalyticalEngine()
    price = engine.price(option, env)
    result = engine.get_last_result()
    return price, result.expected_accrual_ratio, result.per_observation_probs


# =============================================================================
# GATE REPORT
# =============================================================================

def main():
    print("=" * 80)
    print("RANGE ACCRUAL ANALYTICAL ENGINE - INDEPENDENT VERIFICATION")
    print("SR 11-7 Model Validation: Developer B Independent Implementation")
    print("=" * 80)
    print()

    # Run independent implementation
    cases_independent = {
        "Case 1 (Standard 12M monthly)": run_test_case_1,
        "Case 2 (Reverse mode)": run_test_case_2,
        "Case 3 (Narrow range, low vol)": run_test_case_3,
        "Case 4 (Time-varying barriers)": run_test_case_4,
        "Case 5 (Past + future obs)": run_test_case_5,
    }

    cases_dev_a = {
        "Case 1 (Standard 12M monthly)": dev_a_case_1,
        "Case 2 (Reverse mode)": dev_a_case_2,
        "Case 3 (Narrow range, low vol)": dev_a_case_3,
        "Case 4 (Time-varying barriers)": dev_a_case_4,
        "Case 5 (Past + future obs)": dev_a_case_5,
    }

    results = []
    all_pass = True
    tolerance = 1e-10

    for case_name in cases_independent:
        print(f"--- {case_name} ---")

        # Independent implementation
        ind_price, ind_ratio, ind_probs = cases_independent[case_name]()
        print(f"  Independent: price={ind_price:.12f}, ratio={ind_ratio:.12f}")
        print(f"  Independent probs: {[f'{p:.10f}' for p in ind_probs]}")

        # Developer A implementation
        dev_a_price, dev_a_ratio, dev_a_probs = cases_dev_a[case_name]()
        print(f"  Developer A: price={dev_a_price:.12f}, ratio={dev_a_ratio:.12f}")
        print(f"  Developer A probs: {[f'{p:.10f}' for p in dev_a_probs]}")

        # Compare
        abs_diff_price = abs(ind_price - dev_a_price)
        rel_diff_price = abs_diff_price / max(abs(ind_price), 1e-15) if ind_price != 0 else abs_diff_price

        abs_diff_ratio = abs(ind_ratio - dev_a_ratio)
        rel_diff_ratio = abs_diff_ratio / max(abs(ind_ratio), 1e-15) if ind_ratio != 0 else abs_diff_ratio

        case_pass = rel_diff_price < tolerance and rel_diff_ratio < tolerance

        # Also check per-observation probabilities
        prob_max_rel_diff = 0.0
        if len(ind_probs) == len(dev_a_probs):
            for ip, dp in zip(ind_probs, dev_a_probs):
                pd = abs(ip - dp)
                pr = pd / max(abs(ip), 1e-15) if ip != 0 else pd
                prob_max_rel_diff = max(prob_max_rel_diff, pr)
            if prob_max_rel_diff >= tolerance:
                case_pass = False
        else:
            case_pass = False
            prob_max_rel_diff = float('inf')

        if not case_pass:
            all_pass = False

        status = "PASS" if case_pass else "FAIL"
        print(f"  Price diff: abs={abs_diff_price:.2e}, rel={rel_diff_price:.2e}")
        print(f"  Ratio diff: abs={abs_diff_ratio:.2e}, rel={rel_diff_ratio:.2e}")
        print(f"  Prob max rel diff: {prob_max_rel_diff:.2e}")
        print(f"  Status: {status}")
        print()

        results.append({
            "name": case_name,
            "ind_price": ind_price,
            "dev_a_price": dev_a_price,
            "abs_diff": abs_diff_price,
            "rel_diff": rel_diff_price,
            "ind_ratio": ind_ratio,
            "dev_a_ratio": dev_a_ratio,
            "ratio_rel_diff": rel_diff_ratio,
            "prob_max_rel_diff": prob_max_rel_diff,
            "pass": case_pass,
        })

    # Supplementary check: Case 1 + Case 2 ratios should sum to 1.0
    print("--- Supplementary: Standard + Reverse = 1.0 ---")
    c1_ratio = results[0]["ind_ratio"]
    c2_ratio = results[1]["ind_ratio"]
    sum_ratios = c1_ratio + c2_ratio
    supplement_pass = abs(sum_ratios - 1.0) < tolerance
    print(f"  Case 1 ratio: {c1_ratio:.12f}")
    print(f"  Case 2 ratio: {c2_ratio:.12f}")
    print(f"  Sum: {sum_ratios:.15f}")
    print(f"  |sum - 1.0| = {abs(sum_ratios - 1.0):.2e}")
    print(f"  Status: {'PASS' if supplement_pass else 'FAIL'}")
    if not supplement_pass:
        all_pass = False
    print()

    # Research report benchmark check
    print("--- Research Report Benchmark (Case 1) ---")
    print(f"  Research report: E[ratio] = 0.5548, Price = 2.6386")
    print(f"  Independent:     E[ratio] = {results[0]['ind_ratio']:.4f}, Price = {results[0]['ind_price']:.4f}")
    ratio_match = abs(results[0]['ind_ratio'] - 0.5548) < 0.0005
    price_match = abs(results[0]['ind_price'] - 2.6386) < 0.0005
    print(f"  Ratio match (4dp): {'YES' if ratio_match else 'NO'}")
    print(f"  Price match (4dp): {'YES' if price_match else 'NO'}")
    print()

    # GATE REPORT
    print("=" * 80)
    print("GATE REPORT SUMMARY")
    print("=" * 80)
    print()
    header = f"{'Case':<35} {'Ind Price':>14} {'DevA Price':>14} {'Abs Diff':>12} {'Rel Diff':>12} {'Status':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"{r['name']:<35} {r['ind_price']:>14.8f} {r['dev_a_price']:>14.8f} "
            f"{r['abs_diff']:>12.2e} {r['rel_diff']:>12.2e} {status:>8}"
        )
    print()

    overall = "PASS" if all_pass else "FAIL"
    print(f"OVERALL GATE DECISION: {overall}")
    print(f"Tolerance: {tolerance:.0e} (relative)")
    print()

    if all_pass:
        print("Developer A's implementation matches the independent derivation")
        print("to machine precision across all test cases.")
    else:
        print("DISCREPANCIES DETECTED - Review required before production deployment.")
        for r in results:
            if not r["pass"]:
                print(f"  FAILED: {r['name']} (rel_diff={r['rel_diff']:.2e})")

    print()

    return all_pass, results


if __name__ == "__main__":
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    passed, results = main()
    sys.exit(0 if passed else 1)
