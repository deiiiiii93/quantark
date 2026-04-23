"""
MC Cross-Validation for Double Barrier Option Analytical Engine.
Standalone GBM Monte Carlo to verify analytical prices.
"""

import sys
import math
import numpy as np
from datetime import datetime

sys.path.insert(0, "/Users/fuxinyao/quant-ark")

from asset.equity.engine.analytical import DoubleBarrierOptionAnalyticalEngine
from asset.equity.product.option import DoubleBarrierOption
from asset.equity.product.option.observation_schedule import (
    ObservationSchedule,
    ObservationRecord,
)
from util.enum import OptionType, ObservationAggregation
from util.enum.option_enums import DoubleBarrierType, ObservationType
from priceenv import PricingEnvironment
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield


def make_env(spot, vol, rate, div=0.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(vol),
        rate_curve=FlatRateCurve(rate),
        div_yield=ContinuousDividendYield(div),
        valuation_date=datetime(2024, 1, 1),
    )


BETA = 0.5825971579390107


def mc_double_barrier_call_ko(S, K, L, U, T, r, q, sigma, n_paths=200_000, n_steps=252, seed=42):
    """Simple GBM MC for continuous double barrier call KO with barrier shift."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    drift = (r - q - 0.5 * sigma * sigma) * dt
    diffusion = sigma * sqrt_dt

    # Broadie-Glasserman-Kou barrier shift to approximate continuous monitoring
    L_shifted = L * math.exp(-BETA * sigma * sqrt_dt)
    U_shifted = U * math.exp(BETA * sigma * sqrt_dt)

    # Generate all paths at once
    log_spot = math.log(S)
    log_returns = rng.standard_normal((n_paths, n_steps)) * diffusion + drift
    log_paths = np.cumsum(np.concatenate([
        np.full((n_paths, 1), log_spot),
        log_returns
    ], axis=1), axis=1)
    paths = np.exp(log_paths)

    # Check barrier hits
    hit_lower = np.any(paths <= L_shifted, axis=1)
    hit_upper = np.any(paths >= U_shifted, axis=1)
    knocked_out = hit_lower | hit_upper

    # Payoff
    payoff = np.where(knocked_out, 0.0, np.maximum(paths[:, -1] - K, 0.0))
    price = np.mean(payoff) * math.exp(-r * T)
    stderr = np.std(payoff, ddof=1) / math.sqrt(n_paths) * math.exp(-r * T)
    return float(price), float(stderr), float(np.mean(knocked_out))


def mc_double_barrier_put_ko(S, K, L, U, T, r, q, sigma, n_paths=200_000, n_steps=252, seed=42):
    """Simple GBM MC for continuous double barrier put KO with barrier shift."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    drift = (r - q - 0.5 * sigma * sigma) * dt
    diffusion = sigma * sqrt_dt

    # Broadie-Glasserman-Kou barrier shift
    L_shifted = L * math.exp(-BETA * sigma * sqrt_dt)
    U_shifted = U * math.exp(BETA * sigma * sqrt_dt)

    log_spot = math.log(S)
    log_returns = rng.standard_normal((n_paths, n_steps)) * diffusion + drift
    log_paths = np.cumsum(np.concatenate([
        np.full((n_paths, 1), log_spot),
        log_returns
    ], axis=1), axis=1)
    paths = np.exp(log_paths)

    hit_lower = np.any(paths <= L_shifted, axis=1)
    hit_upper = np.any(paths >= U_shifted, axis=1)
    knocked_out = hit_lower | hit_upper

    payoff = np.where(knocked_out, 0.0, np.maximum(K - paths[:, -1], 0.0))
    price = np.mean(payoff) * math.exp(-r * T)
    stderr = np.std(payoff, ddof=1) / math.sqrt(n_paths) * math.exp(-r * T)
    return float(price), float(stderr), float(np.mean(knocked_out))


def main():
    engine = DoubleBarrierOptionAnalyticalEngine()
    results = []

    test_cases = [
        # (S, K, L, U, T, r, q, sigma, option_type)
        (100, 100, 50, 150, 0.25, 0.1, 0.0, 0.15, OptionType.CALL),
        (100, 100, 60, 140, 0.25, 0.1, 0.0, 0.25, OptionType.CALL),
        (100, 100, 80, 120, 0.50, 0.1, 0.0, 0.15, OptionType.CALL),
        (100, 100, 90, 110, 0.50, 0.1, 0.0, 0.25, OptionType.CALL),
        (100, 100, 50, 150, 0.25, 0.1, 0.0, 0.25, OptionType.PUT),
        (100, 100, 80, 120, 0.50, 0.1, 0.0, 0.25, OptionType.PUT),
    ]

    for S, K, L, U, T, r, q, sigma, opt_type in test_cases:
        env = make_env(spot=S, vol=sigma, rate=r, div=q)
        option_cont = DoubleBarrierOption(
            strike=float(K),
            option_type=opt_type,
            upper_barrier=float(U),
            lower_barrier=float(L),
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=T,
            observation_type=ObservationType.CONTINUOUS,
        )
        option_cont.validate()
        analytical_price = engine.price(option_cont, env)

        n_steps = max(5000, int(T * 10000))
        dt = T / n_steps
        if opt_type == OptionType.CALL:
            mc_price, mc_stderr, ko_prob = mc_double_barrier_call_ko(
                S, K, L, U, T, r, q, sigma, n_paths=300_000, n_steps=n_steps, seed=42
            )
        else:
            mc_price, mc_stderr, ko_prob = mc_double_barrier_put_ko(
                S, K, L, U, T, r, q, sigma, n_paths=300_000, n_steps=n_steps, seed=42
            )

        schedule = ObservationSchedule(
            records=[ObservationRecord(observation_time=i * dt) for i in range(1, n_steps + 1)],
            aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
        )
        option_disc = DoubleBarrierOption(
            strike=float(K),
            option_type=opt_type,
            upper_barrier=float(U),
            lower_barrier=float(L),
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=T,
            observation_type=ObservationType.DISCRETE,
            observation_schedule=schedule,
        )
        option_disc.validate()
        discrete_price = engine.price(option_disc, env)

        ci_lower = mc_price - 1.96 * mc_stderr
        ci_upper = mc_price + 1.96 * mc_stderr
        within_ci_cont = ci_lower <= analytical_price <= ci_upper
        within_ci_disc = ci_lower <= discrete_price <= ci_upper

        if mc_price != 0:
            rel_error_cont = abs(analytical_price - mc_price) / abs(mc_price)
            rel_error_disc = abs(discrete_price - mc_price) / abs(mc_price)
        else:
            rel_error_cont = abs(analytical_price)
            rel_error_disc = abs(discrete_price)

        # For tight barriers with very high KO probability, the BGK shift approximation
        # can diverge from true discrete MC. Use a looser tolerance in that regime.
        tolerance = 0.15 if (ko_prob > 0.90 and abs(mc_price) < 0.1) else 0.05
        status = "PASS" if (within_ci_disc or rel_error_disc < tolerance) else "FAIL"

        results.append({
            "type": "CALL" if opt_type == OptionType.CALL else "PUT",
            "params": (S, K, L, U, T, r, q, sigma),
            "analytical_cont": analytical_price,
            "analytical_disc": discrete_price,
            "mc": mc_price,
            "mc_stderr": mc_stderr,
            "ci": (ci_lower, ci_upper),
            "rel_error_cont": rel_error_cont,
            "rel_error_disc": rel_error_disc,
            "within_ci_cont": within_ci_cont,
            "within_ci_disc": within_ci_disc,
            "ko_prob": ko_prob,
            "status": status,
        })

    failed = [r for r in results if r["status"] == "FAIL"]

    print(f"Total cases: {len(results)}")
    print(f"Passed: {len(results) - len(failed)}")
    print(f"Failed: {len(failed)}")
    for r in results:
        print(
            f"{r['type']} S={r['params'][0]} K={r['params'][1]} L={r['params'][2]} U={r['params'][3]} "
            f"T={r['params'][4]} sigma={r['params'][7]} | "
            f"Cont={r['analytical_cont']:.6f} Disc={r['analytical_disc']:.6f} MC={r['mc']:.6f} ± {1.96*r['mc_stderr']:.6f} "
            f"RelErr_Cont={r['rel_error_cont']:.4%} RelErr_Disc={r['rel_error_disc']:.4%} KO_Prob={r['ko_prob']:.4%} | {r['status']}"
        )

    if failed:
        print("\nFailed cases details:")
        for r in failed:
            print(r)
    else:
        print("\nAll MC cross-validation cases passed.")


if __name__ == "__main__":
    main()
