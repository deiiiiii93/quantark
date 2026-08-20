"""Candidate 3: hoist the transcendentals out of the snowball bridge-KI loop.

_check_ki_barriers_continuous_with_bridge is 59% of a continuous-KI snowball
pricing (prof_baseline.py). Its per-step loop re-runs safe_log/exp/clip on
small subsets, plus per-step nonzero/any — ~1000 small NumPy calls with Python
dispatch overhead.

The hoist precomputes the node logs, crossing probabilities, and breach masks
in three full-matrix operations, then keeps the ORIGINAL sequential loop for
the RNG and bookkeeping. The RNG stream is untouched by construction: the same
`rng.random(idx.size)` calls fire in the same order with the same sizes,
because candidate selection depends only on values that are precomputed
bit-identically. Numba was rejected for this loop: draws are data-dependent
(`rng.random(idx.size)` per step) and numba cannot own a Generator stream
bit-compatibly.

Run:  PYTHONPATH=$PWD <venv>/bin/python docs/mc1d-perf/demo_snowball_bridge_hoist.py
"""

import time
from datetime import datetime

import numpy as np

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig, PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import (
    ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ProtectionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_log


SHIPPED_BRIDGE = SnowballMCEngine._check_ki_barriers_continuous_with_bridge


def bridge_hoisted(self, paths, all_times, ki_barrier, sigma, is_reverse, rng_seed):
    """Shipped bridge check with the transcendentals hoisted out of the loop."""
    if ki_barrier <= 0:
        raise ValidationError(f"ki_barrier must be positive, got {ki_barrier}")
    if sigma <= 0:
        raise ValidationError(f"volatility must be positive, got {sigma}")

    n_paths = len(paths)
    if n_paths == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)

    ki_triggered = np.zeros(n_paths, dtype=bool)
    first_ki_idx = np.full(n_paths, -1, dtype=int)

    spot0 = paths[:, 0]
    if is_reverse:
        already_breached = spot0 >= ki_barrier
    else:
        already_breached = spot0 <= ki_barrier
    if already_breached.any():
        ki_triggered[already_breached] = True
        first_ki_idx[already_breached] = 0

    all_times = np.asarray(all_times, dtype=float)
    if all_times.ndim != 1:
        raise ValidationError("all_times must be a 1D array of time points")

    n_steps = paths.shape[1] - 1
    if all_times.shape[0] != n_steps:
        raise ValidationError(
            f"all_times length ({all_times.shape[0]}) must match number of steps ({n_steps})"
        )

    dt = np.empty(n_steps, dtype=float)
    dt[0] = float(all_times[0])
    if n_steps > 1:
        dt[1:] = np.diff(all_times)
    if np.any(dt <= 0.0):
        raise ValidationError("all_times must be strictly increasing and > 0")

    rng = np.random.default_rng(int(rng_seed))

    # ---- hoisted precompute: three full-matrix ops replace ~1000 small ones ----
    left = paths[:, :-1]
    right = paths[:, 1:]
    if is_reverse:
        breach_right = right >= ki_barrier
        non_breached = (left < ki_barrier) & (right < ki_barrier)
    else:
        breach_right = right <= ki_barrier
        non_breached = (left > ki_barrier) & (right > ki_barrier)
    log_nodes = safe_log(paths / ki_barrier)          # (n_paths, n_steps + 1)
    log_term = log_nodes[:, :-1] * log_nodes[:, 1:]   # (n_paths, n_steps)
    h2 = float(sigma * sigma) * dt                    # (n_steps,)
    exponent = np.clip(-2.0 * log_term / h2, -745.0, 0.0)
    p_all = np.exp(exponent)
    # ---------------------------------------------------------------------------

    for k in range(n_steps):
        active = ~ki_triggered
        if not active.any():
            break

        new_hit = active & breach_right[:, k]
        if new_hit.any():
            ki_triggered[new_hit] = True
            first_ki_idx[new_hit] = k

        active = ~ki_triggered
        if not active.any():
            break

        bridge_candidates = active & non_breached[:, k]
        if not bridge_candidates.any():
            continue

        idx = np.flatnonzero(bridge_candidates)
        u = rng.random(idx.size)
        hit = u < p_all[idx, k]
        if hit.any():
            hit_paths = idx[hit]
            ki_triggered[hit_paths] = True
            first_ki_idx[hit_paths] = k

    return ki_triggered, first_ki_idx


def sample_paths(n_paths, n_steps, seed, drifty=-0.15):
    """Paths that visit the KI region often enough to exercise every branch."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    dt = 1.0 / n_steps
    logs = np.cumsum((drifty - 0.02) * dt + 0.35 * np.sqrt(dt) * z, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = 100.0
    paths[:, 1:] = 100.0 * np.exp(logs)
    return paths


def check_bitwise_method():
    n_fail = 0
    engine = SnowballMCEngine(params=MCParams(num_paths=1000, seed=1))
    times = None
    for n_paths, n_steps in ((7, 5), (1024, 252), (100_000, 252)):
        times = np.linspace(1.0 / n_steps, 1.0, n_steps)
        for is_reverse, barrier in ((False, 75.0), (False, 99.0), (True, 130.0)):
            for seed in (0, 42, 20260810):
                paths = sample_paths(n_paths, n_steps, seed,
                                     drifty=0.15 if is_reverse else -0.15)
                a1, a2 = SHIPPED_BRIDGE(engine, paths, times, barrier, 0.35,
                                        is_reverse, seed)
                b1, b2 = bridge_hoisted(engine, paths, times, barrier, 0.35,
                                        is_reverse, seed)
                if a1.tobytes() != b1.tobytes() or a2.tobytes() != b2.tobytes():
                    n_fail += 1
                    print(f"  MISMATCH n={n_paths} steps={n_steps} rev={is_reverse} "
                          f"B={barrier} seed={seed}: "
                          f"trig diff={int(np.sum(a1 != b1))} idx diff={int(np.sum(a2 != b2))}")
    return n_fail


def bench_method():
    engine = SnowballMCEngine(params=MCParams(num_paths=1000, seed=1))
    print("\n  bridge check microbench (best of 5):")
    for n_paths in (8192, 100_000):
        n_steps = 252
        times = np.linspace(1.0 / n_steps, 1.0, n_steps)
        paths = sample_paths(n_paths, n_steps, 3)
        t_ship = t_fast = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            SHIPPED_BRIDGE(engine, paths, times, 75.0, 0.35, False, 42)
            t_ship = min(t_ship, time.perf_counter() - t0)
            t0 = time.perf_counter()
            bridge_hoisted(engine, paths, times, 75.0, 0.35, False, 42)
            t_fast = min(t_fast, time.perf_counter() - t0)
        print(f"    n_paths={n_paths:>7}: shipped {t_ship * 1e3:8.2f} ms   "
              f"hoisted {t_fast * 1e3:8.2f} ms   speedup {t_ship / t_fast:5.2f}x")


def bench_engine():
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )
    product = SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[i / 12 for i in range(1, 13)],
            ki_barrier=75.0, ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
        payoff_config=PayoffConfig(
            rebate_rate=0.15, call_rebate_enabled=False, call_strike=None,
            call_participation_rate=1.0, include_principal=False,
            participation_rate=1.0, protection_type=ProtectionType.NONE,
            protection_rate=0.0,
        ),
        contract_multiplier=10_000.0, maturity=1.0, is_reverse=False,
    )
    results = {}
    print("\n  end-to-end SnowballMCEngine (100k x 252, continuous KI, best of 3):")
    for label, impl in (("shipped", SHIPPED_BRIDGE), ("hoisted", bridge_hoisted)):
        SnowballMCEngine._check_ki_barriers_continuous_with_bridge = impl
        try:
            best = float("inf")
            price = None
            for _ in range(3):
                engine = SnowballMCEngine(
                    params=MCParams(num_paths=100_000, time_steps=252, seed=42),
                    method=MonteCarloMethod.PSEUDO,
                )
                t0 = time.perf_counter()
                price = engine.price(product, env)
                best = min(best, time.perf_counter() - t0)
        finally:
            SnowballMCEngine._check_ki_barriers_continuous_with_bridge = SHIPPED_BRIDGE
        results[label] = (price, best)
        print(f"    {label:>7}: {best:6.3f}s   price {price:.17g}")
    (p_a, t_a), (p_b, t_b) = results["shipped"], results["hoisted"]
    bit_equal = p_a == p_b and p_a.hex() == p_b.hex()
    print(f"    speedup {t_a / t_b:.2f}x, prices bit-equal: {bit_equal}")
    return bit_equal


if __name__ == "__main__":
    print("Candidate 3: snowball bridge-KI transcendental hoist")
    fails = check_bitwise_method()
    print(f"  method bitwise sweep: {'PASS' if fails == 0 else f'{fails} FAILURES'}")
    bench_method()
    ok = bench_engine()
    print(f"\nVERDICT: bitwise={'yes' if fails == 0 and ok else 'NO'}")
