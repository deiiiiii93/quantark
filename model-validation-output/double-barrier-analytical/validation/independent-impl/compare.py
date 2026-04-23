"""
Comparison script: Developer A engine vs Developer B independent implementation.
Uses Haug Table 4-15 benchmark values as ground truth.
"""

import sys
from datetime import datetime

# Add project root to path so we can import Developer A's code as a black box
sys.path.insert(0, "/Users/fuxinyao/quant-ark")

from asset.equity.engine.analytical import DoubleBarrierOptionAnalyticalEngine
from asset.equity.product.option import DoubleBarrierOption
from priceenv import PricingEnvironment
from util.enum import OptionType
from util.enum.option_enums import DoubleBarrierType, ObservationType
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield

from double_barrier_independent import (
    price_double_barrier_call_ko,
    black_scholes_call,
)


# Benchmark cases from Haug Table 4-15 (flat + curvature)
# Format: (L, U, delta1, delta2, T, sigma, expected)
benchmark_cases = [
    # T = 0.25, sigma = 0.15
    (50, 150, 0, 0, 0.25, 0.15, 4.3515),
    (60, 140, 0, 0, 0.25, 0.15, 4.3505),
    (70, 130, 0, 0, 0.25, 0.15, 4.3139),
    (80, 120, 0, 0, 0.25, 0.15, 3.7516),
    (90, 110, 0, 0, 0.25, 0.15, 1.2055),
    # T = 0.25, sigma = 0.25
    (50, 150, 0, 0, 0.25, 0.25, 6.1644),
    (60, 140, 0, 0, 0.25, 0.25, 5.8500),
    (70, 130, 0, 0, 0.25, 0.25, 4.8293),
    (80, 120, 0, 0, 0.25, 0.25, 2.6387),
    (90, 110, 0, 0, 0.25, 0.25, 0.3098),
    # T = 0.5, sigma = 0.15
    (50, 150, 0, 0, 0.50, 0.15, 6.9853),
    (60, 140, 0, 0, 0.50, 0.15, 6.8082),
    (70, 130, 0, 0, 0.50, 0.15, 5.9697),
    (80, 120, 0, 0, 0.50, 0.15, 3.5805),
    (90, 110, 0, 0, 0.50, 0.15, 0.5537),
    # T = 0.5, sigma = 0.25
    (50, 150, 0, 0, 0.50, 0.25, 7.9336),
    (60, 140, 0, 0, 0.50, 0.25, 6.3383),
    (70, 130, 0, 0, 0.50, 0.25, 4.0004),
    (80, 120, 0, 0, 0.50, 0.25, 1.5098),
    (90, 110, 0, 0, 0.50, 0.25, 0.0441),
    # Curvature: delta1=-0.1, delta2=0.1, T=0.25
    (50, 150, -0.1, 0.1, 0.25, 0.15, 4.3514),
    (60, 140, -0.1, 0.1, 0.25, 0.15, 4.3478),
    (70, 130, -0.1, 0.1, 0.25, 0.15, 4.2558),
    (80, 120, -0.1, 0.1, 0.25, 0.15, 3.2953),
    (90, 110, -0.1, 0.1, 0.25, 0.15, 0.5887),
    (50, 150, -0.1, 0.1, 0.25, 0.25, 6.0997),
    (60, 140, -0.1, 0.1, 0.25, 0.25, 5.6351),
    (70, 130, -0.1, 0.1, 0.25, 0.25, 4.3291),
    (80, 120, -0.1, 0.1, 0.25, 0.25, 1.9868),
    (90, 110, -0.1, 0.1, 0.25, 0.25, 0.1016),
    # Curvature: delta1=0.1, delta2=-0.1, T=0.25
    (50, 150, 0.1, -0.1, 0.25, 0.15, 4.3515),
    (60, 140, 0.1, -0.1, 0.25, 0.15, 4.3512),
    (70, 130, 0.1, -0.1, 0.25, 0.15, 4.3382),
    (80, 120, 0.1, -0.1, 0.25, 0.15, 4.0428),
    (90, 110, 0.1, -0.1, 0.25, 0.15, 1.9229),
    (50, 150, 0.1, -0.1, 0.25, 0.25, 6.2040),
    (60, 140, 0.1, -0.1, 0.25, 0.25, 5.9998),
    (70, 130, 0.1, -0.1, 0.25, 0.25, 5.2358),
    (80, 120, 0.1, -0.1, 0.25, 0.25, 3.2872),
    (90, 110, 0.1, -0.1, 0.25, 0.25, 0.6451),
]


def make_env(spot, vol, rate, div=0.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(vol),
        rate_curve=FlatRateCurve(rate),
        div_yield=ContinuousDividendYield(div),
        valuation_date=datetime(2024, 1, 1),
    )


def main():
    engine_a = DoubleBarrierOptionAnalyticalEngine()
    results = []
    max_abs_error_a = 0.0
    max_abs_error_b = 0.0
    max_rel_error_ab = 0.0

    for i, (L, U, delta1, delta2, T, sigma, expected) in enumerate(benchmark_cases, 1):
        env = make_env(spot=100.0, vol=sigma, rate=0.1, div=0.0)
        option = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=float(U),
            lower_barrier=float(L),
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=T,
            observation_type=ObservationType.CONTINUOUS,
        )
        option.validate()

        # Developer A price (black box)
        if delta1 != 0.0 or delta2 != 0.0:
            price_a = engine_a._price_continuous(
                product=option,
                pricing_env=env,
                S=100.0,
                K=100.0,
                T=T,
                r=0.1,
                q=0.0,
                sigma=sigma,
                L=float(L),
                U=float(U),
                multiplier=1.0,
                delta1=delta1,
                delta2=delta2,
            )
        else:
            price_a = engine_a.price(option, env)

        # Developer B price
        price_b = price_double_barrier_call_ko(
            S=100.0,
            K=100.0,
            L=float(L),
            U=float(U),
            T=T,
            r=0.1,
            q=0.0,
            sigma=sigma,
            delta1=delta1,
            delta2=delta2,
            max_terms=10,
        )

        err_a = abs(price_a - expected)
        err_b = abs(price_b - expected)
        if expected != 0:
            rel_err_a = err_a / abs(expected)
            rel_err_b = err_b / abs(expected)
        else:
            rel_err_a = err_a
            rel_err_b = err_b

        # A vs B relative error
        if abs(price_b) > 1e-12:
            rel_err_ab = abs(price_a - price_b) / abs(price_b)
        else:
            rel_err_ab = abs(price_a - price_b)

        max_abs_error_a = max(max_abs_error_a, err_a)
        max_abs_error_b = max(max_abs_error_b, err_b)
        max_rel_error_ab = max(max_rel_error_ab, rel_err_ab)

        status = "PASS" if (err_a <= 1e-3 and err_b <= 1e-3 and rel_err_ab <= 1e-6) else "FAIL"

        results.append({
            "case": i,
            "params": (L, U, delta1, delta2, T, sigma),
            "expected": expected,
            "price_a": price_a,
            "price_b": price_b,
            "err_a": err_a,
            "err_b": err_b,
            "rel_err_ab": rel_err_ab,
            "status": status,
        })

    failed = [r for r in results if r["status"] == "FAIL"]

    print(f"Total cases: {len(results)}")
    print(f"Passed: {len(results) - len(failed)}")
    print(f"Failed: {len(failed)}")
    print(f"Max abs error (Dev A vs benchmark): {max_abs_error_a:.6e}")
    print(f"Max abs error (Dev B vs benchmark): {max_abs_error_b:.6e}")
    print(f"Max relative error (Dev A vs Dev B): {max_rel_error_ab:.6e}")

    if failed:
        print("\nFailed cases:")
        for r in failed:
            print(f"  Case {r['case']}: params={r['params']} expected={r['expected']:.4f} "
                  f"A={r['price_a']:.6f} B={r['price_b']:.6f} err_a={r['err_a']:.6e} "
                  f"err_b={r['err_b']:.6e} rel_ab={r['rel_err_ab']:.6e}")
    else:
        print("\nAll cases passed.")

    # Knock-in parity check (Dev A only, Dev B vanilla helper)
    print("\n--- Knock-In Parity Check (Dev A vs Vanilla - Dev B) ---")
    env = make_env(spot=100.0, vol=0.25, rate=0.1, div=0.0)
    ko = DoubleBarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        upper_barrier=150.0,
        lower_barrier=50.0,
        barrier_type=DoubleBarrierType.KNOCK_OUT,
        maturity=0.25,
        observation_type=ObservationType.CONTINUOUS,
    )
    ki = DoubleBarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        upper_barrier=150.0,
        lower_barrier=50.0,
        barrier_type=DoubleBarrierType.KNOCK_IN,
        maturity=0.25,
        observation_type=ObservationType.CONTINUOUS,
    )
    ko.validate()
    ki.validate()
    ko_price_a = engine_a.price(ko, env)
    ki_price_a = engine_a.price(ki, env)
    vanilla_price_b = black_scholes_call(100.0, 100.0, 0.25, 0.1, 0.0, 0.25)
    parity_lhs = ki_price_a
    parity_rhs = vanilla_price_b - ko_price_a
    parity_err = abs(parity_lhs - parity_rhs)
    print(f"KI (Dev A) = {parity_lhs:.6f}")
    print(f"Vanilla - KO (Dev B vanilla) = {parity_rhs:.6f}")
    print(f"Parity error = {parity_err:.6e} -- {'PASS' if parity_err <= 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()
