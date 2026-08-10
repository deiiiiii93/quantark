"""Shared Andersen QE variance step, with an optional Numba backend.

The QE variance update is the hottest elementwise block in the SLV Monte Carlo
path (``simulate`` carried ~38% of reference-stack tottime, measured
2026-08-10), and it was written out inline in two engines. Sharing it here means
one definition to accelerate and one to verify.

Numba is an OPTIONAL accelerator, exactly like the compiled Thomas kernel: with
it absent the NumPy reference runs and behaviour is unchanged. The Numba path
performs the same operations per element in the same order, so it is
bit-identical -- which is asserted, not assumed, because NumPy 2.x dispatches
array transcendentals to SIMD implementations that need not agree with the
scalar libm calls a per-element loop makes. Measured on this host they do agree
exactly, and ``test_qe_variance_kernel.py`` pins that.

Measured speedup of the Numba path over NumPy (2026-08-10, arm64):
1024 paths 6.1-6.3x, 8192 paths 3.2-3.4x, across ordinary / sigma-collapse /
low-Feller regimes.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

_DEFAULT_PSI_C = 1.5
_DEFAULT_KMIN = 1e-12


class QEVarianceStep(NamedTuple):
    """One QE variance update and the intermediates its callers consume."""

    m: np.ndarray
    a: np.ndarray
    b: np.ndarray
    beta: np.ndarray
    prob_zero: np.ndarray
    v_np: np.ndarray
    v_bar: np.ndarray
    quad_mask: np.ndarray


def qe_variance_step_numpy(
    var: np.ndarray,
    z_var: np.ndarray,
    u_var: np.ndarray,
    *,
    kappa: float,
    theta: float,
    sigma2: float,
    dt: float,
    psi_c: float = _DEFAULT_PSI_C,
    kmin: float = _DEFAULT_KMIN,
) -> QEVarianceStep:
    """Reference implementation: Andersen (2008) QE, vectorized over paths."""
    exp_kdt = np.exp(-kappa * dt)
    omexp = -np.expm1(-kappa * dt)
    m = theta + (var - theta) * exp_kdt
    if kappa > kmin:
        inv_k = 1.0 / kappa
        s2 = (
            var * sigma2 * exp_kdt * (omexp * inv_k)
            + theta * sigma2 * (omexp * omexp * inv_k) / 2.0
        )
    else:
        s2 = var * sigma2 * dt
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = np.where(m <= 1e-12, 0.0, s2 / (m * m))
    psi = np.maximum(psi, 0.0)

    phi = 2.0 / np.maximum(psi, 1e-16)
    rad = np.maximum(phi * (phi - 1.0), 0.0)
    b = np.sqrt(np.maximum(phi - 1.0 + np.sqrt(rad), 0.0))
    a = m / (1.0 + b * b)
    v_a = a * (b + z_var) * (b + z_var)

    prob_zero = np.clip((psi - 1.0) / (psi + 1.0), 0.0, 0.999999)
    beta = np.maximum((1.0 - prob_zero) / np.maximum(m, kmin), kmin)
    u_clip = np.clip(u_var, 1e-12, 1.0 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_b = np.where(
            u_clip <= prob_zero,
            0.0,
            np.log((1.0 - prob_zero) / (1.0 - u_clip)) / beta,
        )

    v_np = np.maximum(np.where(psi <= psi_c, v_a, v_b), 0.0)
    v_bar = np.maximum(0.5 * (v_np + np.maximum(var, 0.0)), 0.0)
    quad_mask = psi <= psi_c
    return QEVarianceStep(m, a, b, beta, prob_zero, v_np, v_bar, quad_mask)


def _build_numba_kernel():
    """Compile the fused kernel, or return None when Numba is unavailable."""
    try:
        from numba import njit
    except ImportError:  # pragma: no cover - depends on the environment
        return None

    # fastmath stays OFF: it licenses reassociation, which would break
    # bit-identity with the NumPy reference.
    @njit(cache=True, fastmath=False)
    def _kernel(
        var, z_var, u_var, kappa, theta, sigma2, dt, psi_c, kmin,
        m_out, a_out, b_out, beta_out, p_out, vnp_out, vbar_out, mask_out,
    ):  # pragma: no cover - exercised through the dispatcher
        exp_kdt = np.exp(-kappa * dt)
        omexp = -np.expm1(-kappa * dt)
        big_kappa = kappa > kmin
        inv_k = 1.0 / kappa if big_kappa else 0.0
        for j in range(var.shape[0]):
            v = var[j]
            m = theta + (v - theta) * exp_kdt
            if big_kappa:
                s2 = (
                    v * sigma2 * exp_kdt * (omexp * inv_k)
                    + theta * sigma2 * (omexp * omexp * inv_k) / 2.0
                )
            else:
                s2 = v * sigma2 * dt
            psi = 0.0 if m <= 1e-12 else s2 / (m * m)
            if psi < 0.0:
                psi = 0.0

            phi = 2.0 / (psi if psi > 1e-16 else 1e-16)
            rad = phi * (phi - 1.0)
            if rad < 0.0:
                rad = 0.0
            inner = phi - 1.0 + np.sqrt(rad)
            if inner < 0.0:
                inner = 0.0
            b = np.sqrt(inner)
            a = m / (1.0 + b * b)
            z = z_var[j]
            v_a = a * (b + z) * (b + z)

            prob_zero = (psi - 1.0) / (psi + 1.0)
            if prob_zero < 0.0:
                prob_zero = 0.0
            elif prob_zero > 0.999999:
                prob_zero = 0.999999
            m_floor = m if m > kmin else kmin
            beta = (1.0 - prob_zero) / m_floor
            if beta < kmin:
                beta = kmin
            u = u_var[j]
            if u < 1e-12:
                u = 1e-12
            elif u > 1.0 - 1e-12:
                u = 1.0 - 1e-12
            if u <= prob_zero:
                v_b = 0.0
            else:
                v_b = np.log((1.0 - prob_zero) / (1.0 - u)) / beta

            quad = psi <= psi_c
            v_new = v_a if quad else v_b
            if v_new < 0.0:
                v_new = 0.0
            v_old = v if v > 0.0 else 0.0
            v_bar = 0.5 * (v_new + v_old)
            if v_bar < 0.0:
                v_bar = 0.0

            m_out[j] = m
            a_out[j] = a
            b_out[j] = b
            beta_out[j] = beta
            p_out[j] = prob_zero
            vnp_out[j] = v_new
            vbar_out[j] = v_bar
            mask_out[j] = quad

    return _kernel


_NUMBA_KERNEL = _build_numba_kernel()


def qe_backend() -> str:
    """``"numba"`` when the accelerator compiled, else ``"numpy"``."""
    return "numba" if _NUMBA_KERNEL is not None else "numpy"


def _qe_variance_step_numba(
    var, z_var, u_var, *, kappa, theta, sigma2, dt,
    psi_c=_DEFAULT_PSI_C, kmin=_DEFAULT_KMIN,
) -> QEVarianceStep:
    var = np.ascontiguousarray(var, dtype=float)
    z_var = np.ascontiguousarray(z_var, dtype=float)
    u_var = np.ascontiguousarray(u_var, dtype=float)
    n = var.shape[0]
    outs = [np.empty(n, dtype=float) for _ in range(7)]
    mask = np.empty(n, dtype=np.bool_)
    _NUMBA_KERNEL(
        var, z_var, u_var, float(kappa), float(theta), float(sigma2), float(dt),
        float(psi_c), float(kmin), *outs, mask,
    )
    m, a, b, beta, prob_zero, v_np, v_bar = outs
    return QEVarianceStep(m, a, b, beta, prob_zero, v_np, v_bar, mask)


def qe_variance_step(
    var, z_var, u_var, *, kappa, theta, sigma2, dt,
    psi_c=_DEFAULT_PSI_C, kmin=_DEFAULT_KMIN,
) -> QEVarianceStep:
    """One QE variance step, via Numba when available and NumPy otherwise."""
    if _NUMBA_KERNEL is not None:
        return _qe_variance_step_numba(
            var, z_var, u_var, kappa=kappa, theta=theta, sigma2=sigma2, dt=dt,
            psi_c=psi_c, kmin=kmin,
        )
    return qe_variance_step_numpy(
        var, z_var, u_var, kappa=kappa, theta=theta, sigma2=sigma2, dt=dt,
        psi_c=psi_c, kmin=kmin,
    )
