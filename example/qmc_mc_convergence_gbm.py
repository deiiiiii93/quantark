#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QMC vs MC convergence example for GBM/BSM European option pricing.

This script compares the convergence behavior of:
    1) Plain Monte Carlo (pseudorandom normals)
    2) Sobol-based QMC with Brownian bridge
    3) RQMC with multiple scrambled Sobol batches

for a simple at-the-money European call option under the Black–Scholes model.
It generates a log–log plot of absolute pricing error vs total number of paths.
"""

from __future__ import annotations

import os
import sys
from math import exp, log, sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Add project root directory to Python path so asset.* imports work even when
# running this script from the example/ directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from asset.equity.process.bsm.qmc_path_generator import (
    GBMPathGenerator,
    GBMPathGeneratorQMC,
)
from asset.equity.process.bsm.qmc_rqmc_driver import run_rqmc
from asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator,
    SobolNormalGenerator,
)


def bs_european_call_price(
    spot: float, strike: float, rrf: float, div: float, vol: float, T: float
) -> float:
    """Black–Scholes price for a European call option."""
    if T <= 0.0:
        return max(spot - strike, 0.0)
    if vol <= 0.0:
        forward = spot * exp((rrf - div) * T)
        return exp(-rrf * T) * max(forward - strike, 0.0)

    sqrtT = sqrt(T)
    d1 = (log(spot / strike) + (rrf - div + 0.5 * vol * vol) * T) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT
    df_r = exp(-rrf * T)
    df_q = exp(-div * T)
    return spot * df_q * norm.cdf(d1) - strike * df_r * norm.cdf(d2)


def european_call_pricer_factory(strike: float, rrf: float, T: float):
    """
    Build a pricer function compatible with run_rqmc.

    The pricer uses terminal values S_T from the provided paths, applies
    discounting, and returns one payoff per path.
    """

    discount = exp(-rrf * T)

    def pricer(paths: np.ndarray, aux) -> np.ndarray:
        ST = paths[:, -1]
        payoff = np.maximum(ST - strike, 0.0)
        # Apply importance sampling weights if present
        if aux is not None and "weights" in aux:
            payoff = payoff * aux["weights"]
        return discount * payoff

    return pricer


def run_convergence_example() -> None:
    # Model parameters
    spot = 100.0
    strike = 100.0
    vol = 0.2
    rrf = 0.03
    div = 0.02
    T = 1.0

    # Path discretization
    time_steps = 64

    # Path counts to test (powers of two for clean Sobol usage)
    n_total_list = [2**10, 2**11, 2**12, 2**13, 2**14, 2**15]

    # Number of batches for RQMC; total paths = n_batches * n_paths_per_batch
    n_batches_rqmc = 8

    # Reference Black–Scholes price
    bs_price = bs_european_call_price(spot, strike, rrf, div, vol, T)
    print(f"Black–Scholes analytic price: {bs_price:.6f}")

    # Prepare containers for results
    errors_mc = []
    errors_qmc = []
    errors_rqmc = []
    std_mc = []
    std_qmc = []
    std_rqmc = []

    # Shared Sobol generator seeds for reproducibility
    sobol_stream_single = SobolNormalGenerator(base_seed=42, strict_power_of_two=True)
    sobol_stream_rqmc = SobolNormalGenerator(base_seed=42, strict_power_of_two=True)

    for n_total in n_total_list:
        print(f"\n=== Total paths: {n_total} ===")

        # ----- Plain MC -----
        rng_mc = PseudoRandomNormalGenerator(seed=42)
        gen_mc = GBMPathGenerator(
            initial_value=spot,
            vol=vol,
            rrf=rrf,
            div=div,
            maturity=T,
            time_steps=time_steps,
            num_paths=n_total,
            model="bsm",
            random_stream=rng_mc,
            use_brownian_bridge=False,
            vr_config=None,
            is_qmc=False,
        )

        paths_mc, _ = gen_mc.generate_paths()
        ST_mc = paths_mc[:, -1]
        payoff_mc = np.maximum(ST_mc - strike, 0.0)
        discount = exp(-rrf * T)
        price_mc = discount * payoff_mc.mean()
        se_mc = discount * payoff_mc.std(ddof=1) / sqrt(n_total)

        err_mc = abs(price_mc - bs_price)
        errors_mc.append(err_mc)
        std_mc.append(se_mc)

        print(f"MC price:   {price_mc:.6f} ± {se_mc:.6f}, abs error = {err_mc:.6f}")

        # ----- Single-batch QMC (Sobol + Brownian bridge) -----
        gen_qmc = GBMPathGeneratorQMC(
            initial_value=spot,
            vol=vol,
            rrf=rrf,
            div=div,
            maturity=T,
            time_steps=time_steps,
            num_paths=n_total,
            model="bsm",
            random_stream=sobol_stream_single,
            use_brownian_bridge=True,
            vr_config=None,
        )

        paths_qmc, _ = gen_qmc.generate_paths(batch_id=0)
        ST_qmc = paths_qmc[:, -1]
        payoff_qmc = np.maximum(ST_qmc - strike, 0.0)
        price_qmc = discount * payoff_qmc.mean()
        se_qmc = discount * payoff_qmc.std(ddof=1) / sqrt(n_total)

        err_qmc = abs(price_qmc - bs_price)
        errors_qmc.append(err_qmc)
        std_qmc.append(se_qmc)

        print(f"QMC price:  {price_qmc:.6f} ± {se_qmc:.6f}, abs error = {err_qmc:.6f}")

        # ----- RQMC with multiple scrambled batches -----
        n_paths_per_batch = n_total // n_batches_rqmc
        if n_paths_per_batch * n_batches_rqmc != n_total:
            raise ValueError("n_total must be divisible by n_batches_rqmc")

        gen_rqmc = GBMPathGeneratorQMC(
            initial_value=spot,
            vol=vol,
            rrf=rrf,
            div=div,
            maturity=T,
            time_steps=time_steps,
            num_paths=n_paths_per_batch,
            model="bsm",
            random_stream=sobol_stream_rqmc,
            use_brownian_bridge=True,
            vr_config=None,
        )

        pricer = european_call_pricer_factory(strike=strike, rrf=rrf, T=T)

        rqmc_result = run_rqmc(
            pricer_fn=pricer,
            path_generator=gen_rqmc,
            max_batches=n_batches_rqmc,
            target_std=1e-4,
            min_batches=n_batches_rqmc,
        )

        price_rqmc = rqmc_result.price
        se_rqmc = rqmc_result.std_error
        err_rqmc = abs(price_rqmc - bs_price)

        # Sanity check: total paths should equal n_total
        if rqmc_result.total_paths != n_total:
            raise RuntimeError(
                f"RQMC total_paths={rqmc_result.total_paths}, expected {n_total}"
            )

        errors_rqmc.append(err_rqmc)
        std_rqmc.append(se_rqmc)

        print(
            f"RQMC price: {price_rqmc:.6f} ± {se_rqmc:.6f}, "
            f"abs error = {err_rqmc:.6f}, batches = {rqmc_result.batches_used}"
        )

    # Convert results to NumPy arrays for plotting
    n_total_arr = np.array(n_total_list, dtype=float)
    errors_mc_arr = np.array(errors_mc, dtype=float)
    errors_qmc_arr = np.array(errors_qmc, dtype=float)
    errors_rqmc_arr = np.array(errors_rqmc, dtype=float)

    # Reference 1/sqrt(N) line based on MC error at the smallest N
    ref_c = errors_mc_arr[0] * sqrt(n_total_arr[0])
    ref_line = ref_c / np.sqrt(n_total_arr)

    # Plot absolute error convergence
    plt.figure(figsize=(8, 6))
    plt.loglog(n_total_arr, errors_mc_arr, "o-", label="MC (pseudorandom)")
    plt.loglog(n_total_arr, errors_qmc_arr, "s-", label="QMC (Sobol + bridge)")
    plt.loglog(n_total_arr, errors_rqmc_arr, "^-", label="RQMC (8 batches)")
    plt.loglog(n_total_arr, ref_line, "k--", label=r"Reference $\mathcal{O}(N^{-1/2})$")

    plt.xlabel("Total number of paths (log scale)")
    plt.ylabel("Absolute pricing error (log scale)")
    plt.title("GBM European Call: MC vs QMC vs RQMC Convergence")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_name = "example/output/qmc_mc_convergence_gbm.png"
    plt.savefig(out_name)
    plt.close()

    print(f"\nConvergence plot saved to: {out_name}")


if __name__ == "__main__":
    run_convergence_example()
