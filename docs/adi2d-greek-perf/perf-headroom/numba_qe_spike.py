"""Spike: does Numba fusion pay on the QE variance step, and is it bitwise?

The QE variance update is the cleanest fusion candidate in the MC stack: ~20
elementwise temporaries per step, no reductions, arrays in / arrays out. This
script extracts it faithfully from
`quantark/volmodels/heston/mc_kernel.py` (QE branch) as a NumPy reference, ports
it to a Numba njit kernel with an explicit per-path loop, and reports both the
speedup and whether the two agree bit for bit.

The bitwise question is genuine and not decidable by inspection: NumPy 2.x
dispatches array transcendentals (exp/log/sqrt) to SIMD implementations that may
differ from scalar libm by an ULP, which is what Numba's per-element loop calls.

Run: PYTHONPATH=<repo> python docs/adi2d-greek-perf/perf-headroom/numba_qe_spike.py
"""

from __future__ import annotations

import time

import numpy as np

KMIN = 1e-12
PSI_C = 1.5


def qe_variance_step_numpy(v_n, zv, uv, kappa, theta, sigma, dt):
    """Reference: the shipped QE variance update, elementwise on arrays."""
    sigma2 = sigma * sigma
    exp_kdt = np.exp(-kappa * dt)
    omexp = -np.expm1(-kappa * dt)
    m = theta + (v_n - theta) * exp_kdt
    if kappa > KMIN:
        inv_k = 1.0 / kappa
        s2 = (
            v_n * sigma2 * exp_kdt * (omexp * inv_k)
            + theta * sigma2 * (omexp * omexp * inv_k) / 2.0
        )
    else:
        s2 = v_n * sigma2 * dt
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(m <= 1e-12, 0.0, s2 / (m * m))
    psi = np.maximum(psi, 0.0)

    phi = 2.0 / np.maximum(psi, 1e-16)
    rad = np.maximum(phi * (phi - 1.0), 0.0)
    B = np.maximum(phi - 1.0 + np.sqrt(rad), 0.0)
    b = np.sqrt(B)
    a = m / (1.0 + b * b)
    v_a = a * (b + zv) * (b + zv)

    p = np.clip((psi - 1.0) / (psi + 1.0), 0.0, 0.999999)
    beta = np.maximum((1.0 - p) / np.maximum(m, KMIN), KMIN)
    u_clip = np.clip(uv, 1e-12, 1.0 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_b = np.where(u_clip <= p, 0.0, np.log((1.0 - p) / (1.0 - u_clip)) / beta)

    v_np = np.where(psi <= PSI_C, v_a, v_b)
    return np.maximum(v_np, 0.0)


def _build_numba_kernel():
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(cache=False, fastmath=False)
    def kernel(v_n, zv, uv, kappa, theta, sigma, dt, out):
        sigma2 = sigma * sigma
        exp_kdt = np.exp(-kappa * dt)
        omexp = -np.expm1(-kappa * dt)
        inv_k = 1.0 / kappa if kappa > KMIN else 0.0
        for j in range(v_n.shape[0]):
            vn = v_n[j]
            m = theta + (vn - theta) * exp_kdt
            if kappa > KMIN:
                s2 = (
                    vn * sigma2 * exp_kdt * (omexp * inv_k)
                    + theta * sigma2 * (omexp * omexp * inv_k) / 2.0
                )
            else:
                s2 = vn * sigma2 * dt
            psi = 0.0 if m <= 1e-12 else s2 / (m * m)
            if psi < 0.0:
                psi = 0.0
            if psi <= PSI_C:
                denom = psi if psi > 1e-16 else 1e-16
                phi = 2.0 / denom
                rad = phi * (phi - 1.0)
                if rad < 0.0:
                    rad = 0.0
                B = phi - 1.0 + np.sqrt(rad)
                if B < 0.0:
                    B = 0.0
                b = np.sqrt(B)
                a = m / (1.0 + b * b)
                z = zv[j]
                val = a * (b + z) * (b + z)
            else:
                p = (psi - 1.0) / (psi + 1.0)
                if p < 0.0:
                    p = 0.0
                elif p > 0.999999:
                    p = 0.999999
                md = m if m > KMIN else KMIN
                beta = (1.0 - p) / md
                if beta < KMIN:
                    beta = KMIN
                u = uv[j]
                if u < 1e-12:
                    u = 1e-12
                elif u > 1.0 - 1e-12:
                    u = 1.0 - 1e-12
                val = 0.0 if u <= p else np.log((1.0 - p) / (1.0 - u)) / beta
            out[j] = val if val > 0.0 else 0.0
        return out

    return kernel


def main() -> None:
    kernel = _build_numba_kernel()
    if kernel is None:
        print("numba not installed; spike cannot run")
        return

    rng = np.random.default_rng(7)
    # sigma_collapse-like and ordinary-like regimes: psi straddles psi_c in one.
    regimes = {
        "ordinary": dict(kappa=1.5, theta=0.04, sigma=0.5, dt=1.0 / 252.0),
        "sigma_collapse": dict(kappa=3.0, theta=0.00306, sigma=0.00311, dt=1.0 / 252.0),
        "low_feller": dict(kappa=0.6, theta=0.09, sigma=1.4, dt=1.0 / 252.0),
    }
    for n_paths in (1024, 8192):
        print(f"\n--- n_paths = {n_paths} ---")
        for name, prm in regimes.items():
            v_n = np.abs(rng.normal(prm["theta"], prm["theta"] * 0.5, n_paths))
            zv = rng.normal(0.0, 1.0, n_paths)
            uv = rng.random(n_paths)
            ref = qe_variance_step_numpy(v_n, zv, uv, **prm)
            out = np.empty_like(v_n)
            got = kernel(v_n, zv, uv, prm["kappa"], prm["theta"], prm["sigma"],
                         prm["dt"], out).copy()
            bitwise = np.array_equal(ref, got)
            max_ulp_diff = np.max(np.abs(ref - got))

            def bench(f, reps=200):
                best = 1e9
                for _ in range(reps):
                    s = time.perf_counter()
                    f()
                    best = min(best, time.perf_counter() - s)
                return best

            t_np = bench(lambda: qe_variance_step_numpy(v_n, zv, uv, **prm))
            t_nb = bench(
                lambda: kernel(v_n, zv, uv, prm["kappa"], prm["theta"],
                               prm["sigma"], prm["dt"], out)
            )
            branch_b = float(np.mean(ref == 0.0))
            print(
                f"  {name:<15} numpy {t_np * 1e6:8.1f} us  numba {t_nb * 1e6:8.1f} us  "
                f"speedup {t_np / t_nb:5.2f}x  bitwise={bitwise}  "
                f"max|diff|={max_ulp_diff:.2e}  zero-frac={branch_b:.2f}"
            )


if __name__ == "__main__":
    main()
