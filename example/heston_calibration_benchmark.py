"""Benchmark: strike-vectorized vs per-option Heston calibration residuals (WS-B1).

Runs BOTH residual paths in one process on one fixture and prints the speedup ratio, so
the >=10x acceptance gate is reproducible from the artifact. Exits non-zero if <10x.
"""
import sys
import time

import numpy as np

from quantark.volmodels.heston import (
    HestonParams, MarketOption, calibrate_heston, heston_call_price,
)


def _build_options(s0, r, q, true):
    strikes = [70, 80, 90, 100, 110, 120, 130]
    mats = [0.25, 0.5, 1.0, 1.5, 2.0]
    return [MarketOption(K=float(k), T=float(t), price=heston_call_price(s0, k, t, true, r, q))
            for t in mats for k in strikes]


def _time_calibrate(s0, opts, r, q, init):
    t0 = time.perf_counter()
    res = calibrate_heston(s0, opts, r, q, init, target="price", regularize_feller=0.0)
    return time.perf_counter() - t0, res


def _time_per_option_baseline(s0, opts, r, q, init):
    """Baseline: force pre-WS-B1 per-option adaptive pricing inside the objective by
    monkeypatching the vectorized entry point the residual loop calls to price strike by
    strike via the adaptive ``heston_call_price``, on the exact same optimizer trajectory."""
    import quantark.volmodels.heston.calibration as calib

    orig = calib.heston_call_prices_vectorized

    def per_option(s0_, strikes, T, params, r_, carry_, **_):
        return np.array([heston_call_price(s0_, float(k), T, params, r_, carry_, method="lewis")
                         for k in np.asarray(strikes, dtype=float)])

    calib.heston_call_prices_vectorized = per_option
    try:
        t0 = time.perf_counter()
        res = calibrate_heston(s0, opts, r, q, init, target="price", regularize_feller=0.0)
        return time.perf_counter() - t0, res
    finally:
        calib.heston_call_prices_vectorized = orig


def main():
    s0, r, q = 100.0, 0.02, 0.0
    true = HestonParams(v0=0.05, kappa=1.5, theta=0.05, sigma=0.4, rho=-0.6)
    init = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.2)
    opts = _build_options(s0, r, q, true)

    dt_base, res_base = _time_per_option_baseline(s0, opts, r, q, init)
    dt_vec, res_vec = _time_calibrate(s0, opts, r, q, init)
    ratio = dt_base / dt_vec if dt_vec > 0 else float("inf")

    print(f"fixture             : {len(opts)} options ({len(set(o.T for o in opts))} maturities)")
    print(f"per-option baseline : {dt_base:.3f}s  nfev={res_base.nfev}")
    print(f"vectorized (shipped): {dt_vec:.3f}s  nfev={res_vec.nfev}")
    print(f"speedup             : {ratio:.1f}x  (gate: >=10x, target 50x)")
    got = np.array([res_vec.params.v0, res_vec.params.kappa, res_vec.params.theta,
                    res_vec.params.sigma, res_vec.params.rho])
    ref = np.array([res_base.params.v0, res_base.params.kappa, res_base.params.theta,
                    res_base.params.sigma, res_base.params.rho])
    print(f"param max-abs diff  : {np.max(np.abs(got - ref)):.2e}")
    if ratio < 10.0:
        print("FAIL: speedup below 10x gate", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
