"""Candidate 2: Numba-fused GBM path build in GBMPathGenerator.generate_paths.

The path build tail (exp of drift+diffusion, then cumprod) is ~70% of a
European MC pricing (prof_baseline.py) and allocates four (n_paths, n_steps)
temporaries: diffusion, the sum, exp_term, and cumprod's output. A fused
per-path Numba loop performs the SAME operations in the SAME order —
c_k = c_{k-1} * exp(drift_dt[k] + vol[k] * dW[p,k]), path = s0 * c_k, exactly
np.cumprod's left-to-right fold with the s0 multiply applied after the fold,
never folded into the accumulator (FP multiplication is not associative).

Same optional-accelerator contract as quantark/montecarlo/qe_kernels.py:
njit(cache=True, fastmath=False); bit-identity asserted, not assumed.

Run:  PYTHONPATH=$PWD <venv>/bin/python docs/mc1d-perf/demo_gbm_numba_fusion.py
"""

import time
from datetime import datetime

import numpy as np
from numba import njit

from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.process.bsm import qmc_path_generator
from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.asset.equity.process.bsm.qmc_sobol import PseudoRandomNormalGenerator
from quantark.asset.equity.process.bsm.qmc_variance_reduction import (
    apply_variance_reduction_to_normals,
)
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import (
    ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod


@njit(cache=True, fastmath=False)
def _gbm_build_kernel(dW, drift_dt, vol, s0, out):  # pragma: no cover - via demo
    n_paths, n_steps = dW.shape
    for p in range(n_paths):
        c = 1.0
        for k in range(n_steps):
            c = c * np.exp(drift_dt[k] + vol[k] * dW[p, k])
            out[p, k + 1] = s0 * c


def numpy_tail(gen, dW):
    """The shipped path-build tail, verbatim from generate_paths."""
    paths = np.zeros((dW.shape[0], gen.time_steps + 1), dtype=float)
    paths[:, 0] = gen.initial_value
    drift_term = (gen._drift_vec - 0.5 * gen._vol_vec * gen._vol_vec) * gen.dt_vector
    drift_term = drift_term.reshape(1, -1)
    diffusion_term = gen._vol_vec.reshape(1, -1) * dW
    exp_term = np.exp(drift_term + diffusion_term)
    paths[:, 1:] = gen.initial_value * np.cumprod(exp_term, axis=1)
    return paths


def numba_tail(gen, dW):
    paths = np.empty((dW.shape[0], gen.time_steps + 1), dtype=float)
    paths[:, 0] = gen.initial_value
    drift_dt = np.ascontiguousarray(
        (gen._drift_vec - 0.5 * gen._vol_vec * gen._vol_vec) * gen.dt_vector
    )
    _gbm_build_kernel(
        np.ascontiguousarray(dW), drift_dt,
        np.ascontiguousarray(gen._vol_vec), float(gen.initial_value), paths,
    )
    return paths


def make_generator(n_paths, n_steps=252, seed=42, term_structure=True):
    """Pseudo-MC generator; per-step r/q/vol vectors to exercise the term path."""
    if term_structure:
        k = np.arange(n_steps)
        rrf = 0.03 + 0.02 * k / n_steps
        div = 0.01 + 0.01 * k / n_steps
        vol = 0.18 + 0.06 * np.sin(2 * np.pi * k / n_steps) ** 2
    else:
        rrf, div, vol = 0.05, 0.02, 0.2
    return GBMPathGenerator(
        initial_value=100.0, vol=vol, rrf=rrf, div=div, maturity=1.0,
        time_steps=n_steps, num_paths=n_paths, model="bsm",
        random_stream=PseudoRandomNormalGenerator(seed=seed),
        use_brownian_bridge=False, vr_config=None, is_qmc=False,
    )


def draws_for(gen):
    z = gen._generate_base_normals(batch_id=None)
    z, _, _ = apply_variance_reduction_to_normals(
        n_paths=gen.num_paths, dim=gen.time_steps, base_normals=z,
        vr_config=gen.vr_config, is_qmc=gen.is_qmc,
    )
    return gen._build_brownian_increments(z)


def check_bitwise():
    n_fail = 0
    for n_paths, n_steps, term in ((7, 5, True), (1024, 252, True),
                                   (100_000, 252, True), (8192, 63, False)):
        gen = make_generator(n_paths, n_steps, term_structure=term)
        dW = draws_for(gen)
        a = numpy_tail(gen, dW)
        b = numba_tail(gen, dW)
        if a.tobytes() != b.tobytes():
            n_fail += 1
            print(f"  MISMATCH n_paths={n_paths} n_steps={n_steps} "
                  f"max|d|={np.max(np.abs(a - b))}")
    return n_fail


def bench_tail():
    print("\n  path-build tail (best of 5, same dW):")
    for n_paths in (8192, 100_000, 200_000):
        gen = make_generator(n_paths)
        dW = draws_for(gen)
        t_np = t_nb = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            numpy_tail(gen, dW)
            t_np = min(t_np, time.perf_counter() - t0)
            t0 = time.perf_counter()
            numba_tail(gen, dW)
            t_nb = min(t_nb, time.perf_counter() - t0)
        print(f"    n_paths={n_paths:>7}: numpy {t_np * 1e3:8.2f} ms   "
              f"numba {t_nb * 1e3:8.2f} ms   speedup {t_np / t_nb:5.2f}x")


SHIPPED_GENERATE_PATHS = GBMPathGenerator.generate_paths


def generate_paths_patched(self, seed=None, batch_id=None, return_aux=False):
    """Shipped generate_paths with only the build tail swapped for the kernel."""
    if seed is not None and isinstance(self.random_stream, PseudoRandomNormalGenerator):
        self.random_stream = PseudoRandomNormalGenerator(seed=seed)
    base_normals = self._generate_base_normals(batch_id=batch_id)
    z_processed, weights, control_variate = apply_variance_reduction_to_normals(
        n_paths=self.num_paths, dim=self.time_steps, base_normals=base_normals,
        vr_config=self.vr_config, is_qmc=self.is_qmc,
    )
    dW = self._build_brownian_increments(z_processed)
    paths = numba_tail(self, dW)
    aux = None
    if return_aux:
        aux = {"batch_id": np.array(batch_id if batch_id is not None else 0)}
        if weights is not None:
            aux["weights"] = weights
        if control_variate is not None:
            aux["control_variate"] = control_variate
    return paths, aux


def bench_engine():
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )
    product = EuropeanVanillaOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    results = {}
    print("\n  end-to-end EuropeanMCEngine (200k x 252, pseudo, best of 3):")
    for label, impl in (("shipped", SHIPPED_GENERATE_PATHS),
                        ("numba", generate_paths_patched)):
        GBMPathGenerator.generate_paths = impl
        try:
            best = float("inf")
            price = None
            for _ in range(3):
                engine = EuropeanMCEngine(
                    params=MCParams(num_paths=200_000, time_steps=252, seed=42),
                    method=MonteCarloMethod.PSEUDO,
                )
                t0 = time.perf_counter()
                price = engine.price(product, env)
                best = min(best, time.perf_counter() - t0)
        finally:
            GBMPathGenerator.generate_paths = SHIPPED_GENERATE_PATHS
        results[label] = (price, best)
        print(f"    {label:>7}: {best:6.3f}s   price {price:.17g}")
    (p_a, t_a), (p_b, t_b) = results["shipped"], results["numba"]
    bit_equal = p_a == p_b and p_a.hex() == p_b.hex()
    print(f"    speedup {t_a / t_b:.2f}x, prices bit-equal: {bit_equal}")
    return bit_equal


if __name__ == "__main__":
    print("Candidate 2: Numba-fused GBM path build")
    # compile outside all timers
    warm = make_generator(8, 4)
    numba_tail(warm, draws_for(warm))
    fails = check_bitwise()
    print(f"  path bitwise sweep: {'PASS' if fails == 0 else f'{fails} FAILURES'}")
    bench_tail()
    ok = bench_engine()
    print(f"\nVERDICT: bitwise={'yes' if fails == 0 and ok else 'NO'}")
