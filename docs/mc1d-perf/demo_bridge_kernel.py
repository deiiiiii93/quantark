"""Spike: fused Numba kernel for BrownianBridge.transform.

WHY THIS ONE. The otc-price-adapter book (2026-08-11) showed Phase 1 buying
only 1.06x on real desk rows, and the profile said where the time actually
is: of a heavy snowball row, qmc_brownian_bridge.transform is 32% and
qmc_sobol.normal is 29%, while the path build Phase 1 accelerated is 17%.

WHY IT SHOULD WORK. transform loops in Python over n_steps (~500 for these
trades) and each iteration reads/writes COLUMNS -- W[:, l], W[:, r], W[:, k]
-- of a C-contiguous (n_paths, n_steps) array, so consecutive elements of
each vector op are n_steps*8 bytes apart. The fused kernel walks one path at
a time, keeping the bridge recursion in registers with row-major access.

GATE (declared before measuring): dW byte-identical across shapes and time
grids, AND >= 2.0x on the transform at production shape (100k x ~488). 2x on
a 32% share is ~1.19x per production row.

BITWISE ARGUMENT. Every operation here (+, -, *, /, sqrt) is IEEE-754
correctly rounded, so a scalar loop and a SIMD vector op must agree exactly
-- a stronger footing than the path kernel, whose exp() agreement was an
empirical question about libm vs SIMD. Two order traps are respected:
  1. mean keeps the shipped grouping (a*W_l + b*W_r) / denom. Folding
     a/denom into a coefficient would reassociate and is NOT bitwise.
  2. when left == -1 the shipped code still evaluates a*0.0 and ADDS it.
     Skipping that term looks harmless but flips the sign of a zero
     (0.0 + -0.0 == +0.0, while -0.0 alone stays -0.0), which a byte
     comparison catches.

Run:  PYTHONPATH=$PWD <venv>/bin/python docs/mc1d-perf/demo_bridge_kernel.py
"""

import time
from datetime import datetime

import numpy as np
from numba import njit

from quantark.montecarlo.qmc_brownian_bridge import BrownianBridge

SHIPPED_TRANSFORM = BrownianBridge.transform


@njit(cache=True, fastmath=False)
def _bridge_kernel(z, idx, left, right, a, b, denom, stds, W, dW):  # pragma: no cover
    n_paths, n_steps = z.shape
    for p in range(n_paths):
        W[p, idx[0]] = stds[0] * z[p, 0]
        for j in range(1, n_steps):
            k = idx[j]
            l = left[j]
            r = right[j]
            W_l = 0.0 if l == -1 else W[p, l]
            W_r = W[p, r]
            # grouping preserved exactly: (a*W_l + b*W_r) / denom
            mean = (a[j] * W_l + b[j] * W_r) / denom[j]
            W[p, k] = mean + stds[j] * z[p, j]
        prev = W[p, 0]
        dW[p, 0] = prev
        for k in range(1, n_steps):
            cur = W[p, k]
            dW[p, k] = cur - prev
            prev = cur


def transform_fused(self, z: np.ndarray) -> np.ndarray:
    """Candidate transform: same arithmetic, per-path fused, row-major."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("z must be a 2D array of shape (n_paths, n_steps)")
    n_paths, n_steps = z.shape
    if n_steps != self.times.shape[0]:
        raise ValueError(
            f"z has {n_steps} time steps but BrownianBridge is configured "
            f"for {self.times.shape[0]} steps."
        )

    times = self.times
    idx = np.ascontiguousarray(self.indices)
    left = np.ascontiguousarray(self.left)
    right = np.ascontiguousarray(self.right)

    # Per-step scalars, computed exactly as the shipped loop computes them.
    a = np.zeros(n_steps)      # t_r - t_m
    b = np.zeros(n_steps)      # t_m - t_l
    denom = np.ones(n_steps)   # t_r - t_l
    for j in range(1, n_steps):
        l, r, k = left[j], right[j], idx[j]
        t_l = 0.0 if l == -1 else times[l]
        t_r = times[r]
        t_m = times[k]
        d = t_r - t_l
        if d <= 0.0:
            raise ValueError("Invalid Brownian bridge interval length.")
        a[j] = t_r - t_m
        b[j] = t_m - t_l
        denom[j] = d
    stds = np.sqrt(self.variances)

    W = np.empty((n_paths, n_steps), dtype=float)
    dW = np.empty((n_paths, n_steps), dtype=float)
    _bridge_kernel(np.ascontiguousarray(z), idx, left, right, a, b, denom,
                   stds, W, dW)
    return dW


def _grids():
    """Time grids covering uniform, non-uniform, tiny, and production sizes."""
    out = {}
    for n in (1, 2, 3, 7, 63, 252, 488):
        out[f"uniform_{n}"] = np.linspace(1.0 / n, 1.0, n)
    rng = np.random.default_rng(5)
    raw = np.sort(rng.uniform(0.001, 3.0, size=97))
    out["nonuniform_97"] = raw
    # clustered grid: dense early, sparse late (typical of a seasoned trade)
    out["clustered_120"] = np.concatenate([
        np.linspace(0.002, 0.25, 90), np.linspace(0.30, 2.0, 30)
    ])
    return out


def check_bitwise():
    fails = 0
    checks = 0
    for name, times in _grids().items():
        bridge = BrownianBridge.from_time_grid(times)
        n_steps = times.size
        for n_paths, seed in ((1, 0), (5, 1), (1024, 2)):
            rng = np.random.default_rng(seed)
            z = rng.standard_normal((n_paths, n_steps))
            a = SHIPPED_TRANSFORM(bridge, z)
            b = transform_fused(bridge, z)
            checks += 1
            if a.tobytes() != b.tobytes():
                fails += 1
                print(f"  MISMATCH {name} n_paths={n_paths} "
                      f"max|d|={np.max(np.abs(a - b)):.3e}")
    # sign-of-zero trap: all-zero normals make every mean exactly +/-0.0
    for name, times in _grids().items():
        bridge = BrownianBridge.from_time_grid(times)
        z = np.zeros((3, times.size))
        checks += 1
        if SHIPPED_TRANSFORM(bridge, z).tobytes() != transform_fused(bridge, z).tobytes():
            fails += 1
            print(f"  MISMATCH (zero-sign trap) {name}")
    print(f"  bitwise sweep: {checks} cases, "
          f"{'PASS' if fails == 0 else f'{fails} FAILURES'}")
    return fails


def bench_transform():
    print("\n  transform microbench (best of 5):")
    results = {}
    for n_paths, n_steps in ((8192, 252), (100_000, 252), (100_000, 488)):
        times = np.linspace(1.0 / n_steps, 1.0, n_steps)
        bridge = BrownianBridge.from_time_grid(times)
        rng = np.random.default_rng(3)
        z = rng.standard_normal((n_paths, n_steps))
        t_ship = t_fast = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            SHIPPED_TRANSFORM(bridge, z)
            t_ship = min(t_ship, time.perf_counter() - t0)
            t0 = time.perf_counter()
            transform_fused(bridge, z)
            t_fast = min(t_fast, time.perf_counter() - t0)
        results[(n_paths, n_steps)] = t_ship / t_fast
        print(f"    {n_paths:>7} x {n_steps:>3}: shipped {t_ship * 1e3:8.1f} ms   "
              f"fused {t_fast * 1e3:8.1f} ms   speedup {t_ship / t_fast:5.2f}x")
    return results[(100_000, 488)]


def bench_snowball():
    """End-to-end on a QMC snowball shaped like the adapter's book rows."""
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

    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.025),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )
    product = SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[i / 12 for i in range(1, 13)],
            ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
            ki_continuous=False,
            ki_observation_dates=[i / 244.0 for i in range(1, 245)],
        ),
        payoff_config=PayoffConfig(
            rebate_rate=0.12, call_rebate_enabled=False, call_strike=None,
            call_participation_rate=1.0, include_principal=False,
            participation_rate=1.0, protection_type=ProtectionType.NONE,
            protection_rate=0.0,
        ),
        contract_multiplier=10_000.0, maturity=1.0, is_reverse=False,
    )
    print("\n  end-to-end snowball QMC (100k paths x 488 steps, best of 3):")
    out = {}
    for label, impl in (("shipped", SHIPPED_TRANSFORM), ("fused", transform_fused)):
        BrownianBridge.transform = impl
        try:
            best = float("inf")
            price = None
            for _ in range(3):
                engine = SnowballMCEngine(
                    params=MCParams(num_paths=100_000, time_steps=488, seed=42),
                    method=MonteCarloMethod.QUASI,
                )
                t0 = time.perf_counter()
                price = engine.price(product, env)
                best = min(best, time.perf_counter() - t0)
        finally:
            BrownianBridge.transform = SHIPPED_TRANSFORM
        out[label] = (price, best)
        print(f"    {label:>7}: {best:6.3f}s   price {price:.17g}")
    (pa, ta), (pb, tb) = out["shipped"], out["fused"]
    same = pa == pb and pa.hex() == pb.hex()
    print(f"    speedup {ta / tb:.2f}x, prices bit-equal: {same}")
    return same, ta / tb


if __name__ == "__main__":
    print("Spike: fused Brownian-bridge transform")
    warm = BrownianBridge.from_time_grid(np.linspace(0.25, 1.0, 4))
    transform_fused(warm, np.zeros((2, 4)))  # compile outside the timers
    fails = check_bitwise()
    prod_speedup = bench_transform()
    same, e2e = bench_snowball()
    gate = fails == 0 and same and prod_speedup >= 2.0
    print(f"\nVERDICT: bitwise={'yes' if fails == 0 and same else 'NO'}, "
          f"transform {prod_speedup:.2f}x @100k x 488, snowball {e2e:.2f}x")
    print(f"GATE (bitwise AND transform >= 2.0x): {'PASS' if gate else 'FAIL'}")
