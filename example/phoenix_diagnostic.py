"""
Phoenix Engine Diagnostic Test - Systematic Root Cause Analysis

This script isolates which component causes pricing discrepancies by:
1. Turning off KI (set very high)
2. Turning off coupon kink (always paid)
3. Moving KO far away
4. Testing Quad domain/padding/filter variations

Run: python example/phoenix_diagnostic.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from asset.equity.param import MCParams, PDEParams, QuadParams
from asset.equity.product.option import create_standard_phoenix
from asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.enum import CouponPayType, ObservationType


def create_env(spot=100.0, vol=0.20, rate=0.03):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2024, 1, 1),
    )


def build_schedule(times, barrier):
    return ObservationSchedule(
        records=[ObservationRecord(observation_time=t, barrier=barrier) for t in times]
    )


def price_with_all_three(product, env, label=""):
    quad = PhoenixQuadEngine(params=QuadParams(grid_points=401))
    pde = PhoenixPDESolver(params=PDEParams(grid_size=200, time_steps=100))
    mc = PhoenixMCEngine(params=MCParams(num_paths=50000, seed=42))

    q = quad.price(product, env)
    p = pde.price(product, env)
    m = mc.price(product, env)

    diff_qm = abs(q - m) / abs(m) if m != 0 else 0
    diff_pm = abs(p - m) / abs(m) if m != 0 else 0
    diff_qp = abs(q - p) / abs(p) if p != 0 else 0

    print(f"\n{label}")
    print(f"  Quad: {q:>12,.2f} | PDE: {p:>12,.2f} | MC: {m:>12,.2f}")
    print(f"  Q-M: {diff_qm:>7.2%} | P-M: {diff_pm:>7.2%} | Q-P: {diff_qp:>7.2%}")

    return q, p, m, diff_qm, diff_pm, diff_qp


def print_separator(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    env = create_env()
    monthly = [i / 12 for i in range(1, 13)]

    results = {}

    # ========================================================================
    # BASELINE: Standard Phoenix (all features enabled)
    # ========================================================================
    print_separator("BASELINE: Standard Phoenix (all features)")
    ki_monthly = build_schedule(monthly, 75.0)
    q, p, m, *_ = price_with_all_three(
        create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ki_monthly,
        ),
        env,
        "Standard Phoenix (KO=103, KI=75, Coupon=95)",
    )
    results["baseline"] = {"qm": abs(q - m) / abs(m), "pm": abs(p - m) / abs(m), "qp": abs(q - p) / abs(p)}

    # ========================================================================
    # TEST 1: Turn off KI (set very high - should never trigger)
    # Expected: If oscillation decreases, FFT tail treatment is the issue
    # ========================================================================
    print_separator("TEST 1: KI Disabled (KI=999, far above spot)")
    ki_high = build_schedule(monthly, 999.0)
    q, p, m, *_ = price_with_all_three(
        create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=999.0,  # Very high - never triggers
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ki_high,
        ),
        env,
        "KI Disabled (KO=103, KI=999, Coupon=95)",
    )
    results["no_ki"] = {"qm": abs(q - m) / abs(m), "pm": abs(p - m) / abs(m), "qp": abs(q - p) / abs(p)}

    # ========================================================================
    # TEST 2: Turn off coupon kink (coupon always paid)
    # Expected: If oscillation decreases, coupon digital barrier is the issue
    # ========================================================================
    print_separator("TEST 2: Coupon Always Paid (barrier=1, always triggered)")
    ki_monthly = build_schedule(monthly, 75.0)
    q, p, m, *_ = price_with_all_three(
        create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=1.0,  # Always paid (spot >= 1 always true)
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ki_monthly,
        ),
        env,
        "Coupon Always Paid (KO=103, KI=75, Coupon=1)",
    )
    results["coupon_smooth"] = {"qm": abs(q - m) / abs(m), "pm": abs(p - m) / abs(m), "qp": abs(q - p) / abs(p)}

    # ========================================================================
    # TEST 3: Move KO far away (hard to trigger)
    # Expected: If PDE/Quad agree better, near-spot KO is the sensitivity amplifier
    # ========================================================================
    print_separator("TEST 3: KO Far Away (KO=150, above spot)")
    ki_monthly = build_schedule(monthly, 75.0)
    q, p, m, *_ = price_with_all_three(
        create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=150.0,  # Far above spot - rarely triggers
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ki_monthly,
        ),
        env,
        "KO Far Away (KO=150, KI=75, Coupon=95)",
    )
    results["ko_far"] = {"qm": abs(q - m) / abs(m), "pm": abs(p - m) / abs(m), "qp": abs(q - p) / abs(p)}

    # ========================================================================
    # TEST 4: KO very close (always triggers immediately)
    # Expected: All engines should agree (just KO payoff + coupons)
    # ========================================================================
    print_separator("TEST 4: KO Always Triggered (KO=90, below spot)")
    ki_monthly = build_schedule(monthly, 75.0)
    q, p, m, *_ = price_with_all_three(
        create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=90.0,  # Below spot - KO immediately at valuation
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ki_monthly,
        ),
        env,
        "KO Always Triggered (KO=90, KI=75, Coupon=95)",
    )
    results["ko_always"] = {"qm": abs(q - m) / abs(m), "pm": abs(p - m) / abs(m), "qp": abs(q - p) / abs(p)}

    # ========================================================================
    # TEST 5: Reverse Phoenix with KI disabled
    # Expected: Should isolate if the 26% Reverse Phoenix discrepancy is KI-related
    # ========================================================================
    print_separator("TEST 5: Reverse Phoenix with KI Disabled")
    ki_high = build_schedule(monthly, 999.0)
    q, p, m, *_ = price_with_all_three(
        create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=999.0,  # No KI
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ki_high,
        ),
        env,
        "Reverse-like (KO=103, KI=999, Coupon=95) - no KI",
    )
    results["reverse_no_ki"] = {"qm": abs(q - m) / abs(m), "pm": abs(p - m) / abs(m), "qp": abs(q - p) / abs(p)}

    # ========================================================================
    # TEST 6: No barriers at all (pure coupon stream)
    # Expected: Best case - all engines should agree
    # ========================================================================
    print_separator("TEST 6: No Barriers (Pure Coupon Stream)")
    q, p, m, *_ = price_with_all_three(
        create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=999.0,  # Never triggers
            ki_barrier=999.0,  # Never triggers
            coupon_barrier=1.0,  # Always paid
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=build_schedule(monthly, 999.0),
        ),
        env,
        "No Barriers (KO=999, KI=999, Coupon=1)",
    )
    results["no_barriers"] = {"qm": abs(q - m) / abs(m), "pm": abs(p - m) / abs(m), "qp": abs(q - p) / abs(p)}

    # ========================================================================
    # TEST 7: Quad Domain Variations
    # ========================================================================
    print_separator("TEST 7: Quad Domain/Grid Variations")

    ki_monthly = build_schedule(monthly, 75.0)
    phoenix = create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        coupon_barrier=95.0,
        ko_rate=0.12,
        coupon_rate=0.02,
        num_observations=12,
        memory_coupon=False,
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_schedule=ki_monthly,
    )

    print("\n  Quad Grid Points Convergence:")
    for n_points in [201, 301, 401, 501, 601, 801, 1001]:
        quad = PhoenixQuadEngine(params=QuadParams(grid_points=n_points))
        price = quad.price(phoenix, env)
        print(f"    {n_points:4d} points: {price:>12,.2f}")

    print("\n  PDE Grid Convergence:")
    for grid_size in [100, 150, 200, 300, 400]:
        pde = PhoenixPDESolver(params=PDEParams(grid_size=grid_size, time_steps=grid_size // 2))
        price = pde.price(phoenix, env)
        print(f"    {grid_size:4d} grid: {price:>12,.2f}")

    # ========================================================================
    # SUMMARY: What changed?
    # ========================================================================
    print_separator("DIAGNOSTIC SUMMARY")
    print("\n  Relative Differences (%):")
    print(f"  {'Test':<30} {'Q-M':>7} {'P-M':>7} {'Q-P':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7}")

    for name, data in results.items():
        qm = data["qm"] * 100
        pm = data["pm"] * 100
        qp = data["qp"] * 100
        print(f"  {name:<30} {qm:>6.1f}% {pm:>6.1f}% {qp:>6.1f}%")

    print("\n  Diagnosis:")
    baseline_qm = results["baseline"]["qm"] * 100
    no_ki_qm = results["no_ki"]["qm"] * 100
    coupon_smooth_qm = results["coupon_smooth"]["qm"] * 100
    ko_far_qm = results["ko_far"]["qm"] * 100
    no_barriers_qm = results["no_barriers"]["qm"] * 100

    print(f"    Baseline Q-M:        {baseline_qm:.1f}%")
    print(f"    No KI Q-M:           {no_ki_qm:.1f}% (change: {no_ki_qm - baseline_qm:+.1f}%)")
    print(f"    Coupon smooth Q-M:    {coupon_smooth_qm:.1f}% (change: {coupon_smooth_qm - baseline_qm:+.1f}%)")
    print(f"    KO far Q-M:          {ko_far_qm:.1f}% (change: {ko_far_qm - baseline_qm:+.1f}%)")
    print(f"    No barriers Q-M:      {no_barriers_qm:.1f}% (change: {no_barriers_qm - baseline_qm:+.1f}%)")

    print("\n  Interpretation:")
    if abs(no_ki_qm - baseline_qm) < 1:
        print("    ✓ KI contribution: MINIMAL - KI handling is consistent")
    else:
        print(f"    ✗ KI contribution: SIGNIFICANT - {abs(no_ki_qm - baseline_qm):.1f}% change")

    if abs(coupon_smooth_qm - baseline_qm) < 1:
        print("    ✓ Coupon kink contribution: MINIMAL - Digital coupon not the issue")
    else:
        print(f"    ✗ Coupon kink contribution: SIGNIFICANT - {abs(coupon_smooth_qm - baseline_qm):.1f}% change")

    if abs(ko_far_qm - baseline_qm) < 1:
        print("    ✓ KO proximity contribution: MINIMAL - KO location not the issue")
    else:
        print(f"    ✗ KO proximity contribution: SIGNIFICANT - {abs(ko_far_qm - baseline_qm):.1f}% change")

    if no_barriers_qm < 2:
        print("    ✓ With no barriers: All engines EXCELLENT agreement")
    else:
        print(f"    ✗ With no barriers: Still {no_barriers_qm:.1f}% difference - core diffusion issue")


if __name__ == "__main__":
    main()
