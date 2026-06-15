#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SABR implied volatility utilities (Hagan's lognormal approximation).

This module provides numerically stable implementations of the SABR Black
implied volatility approximation, including:
  - General-strike formula with small-z stabilization
  - ATM-forward specialization
  - Shifted-SABR support via simple forward/strike shift

References
----------
Hagan, P., Kumar, D., Lesniewski, A., & Woodward, D. (2002).
Managing Smile Risk.
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np

from quantark.util.numerical import Tolerance

ArrayLike = Union[float, np.ndarray]

# --- Model/algorithm constants (NOT comparison tolerances) -------------------
# Radius around z = 0 where Hagan's chi(z) is evaluated with its Taylor series
# rather than the closed form, to avoid catastrophic cancellation in the log.
_SMALL_Z_SERIES_RADIUS: float = 1e-8
# Below this |z| the ratio x = z / chi(z) is replaced by its exact limit 1.
_X_LIMIT_RADIUS: float = 1e-12
# Exact ATM detection: |F - K| below this falls back to the ATM specialization.
_ATM_GAP: float = 1e-12

# Domain bounds for SABR parameters.
_RHO_BOUND: float = 0.999


def _ensure_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=float)


def _chi_z(z: np.ndarray, rho: float) -> np.ndarray:
    """
    Numerically stable implementation of chi(z) used in Hagan's SABR expansion:
        chi(z) = ln( (sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho) )

    For |z| ~ 0, use a Taylor expansion to avoid cancellation.
    """
    z = _ensure_numpy(z)
    rho = float(rho)
    out = np.empty_like(z)

    small_mask = np.abs(z) < _SMALL_Z_SERIES_RADIUS
    large_mask = ~small_mask

    # Series expansion around z = 0 (l'Hospital):
    #   chi(z) ~= z * (1 - rho*z/2 + (2 - 3*rho^2) * z^2 / 6) for small z
    if np.any(small_mask):
        zs = z[small_mask]
        out[small_mask] = zs * (
            1.0 - rho * zs / 2.0 + (2.0 - 3.0 * rho * rho) * (zs * zs) / 6.0
        )

    if np.any(large_mask):
        zl = z[large_mask]
        numerator = np.sqrt(1.0 - 2.0 * rho * zl + zl * zl) + zl - rho
        denominator = 1.0 - rho
        # Guard against non-positive arguments inside log (extreme inputs).
        ratio = np.maximum(numerator / denominator, Tolerance.LOG_MIN)
        out[large_mask] = np.log(ratio)

    return out


def sabr_implied_vol_black(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
) -> np.ndarray:
    """
    Hagan's lognormal (Black) implied volatility approximation for SABR.

    Parameters
    ----------
    F : float or array
        Forward price at t0. If shift != 0, pass the unshifted forward; the
        function applies the shift internally.
    K : float or array
        Strike(s). If shift != 0, pass unshifted strikes; the function applies
        the shift internally.
    T : float or array
        Time to maturity in years.
    alpha : float
        Initial volatility level (alpha0), strictly positive.
    beta : float
        Elasticity parameter in [0, 1].
    rho : float
        Correlation in [-1, 1].
    nu : float
        Volatility of volatility, strictly positive.
    shift : float, optional
        Additive shift for shifted-SABR (default 0.0). When provided, the
        approximation is evaluated on F+shift and K+shift.

    Returns
    -------
    np.ndarray
        Black implied volatility sigma_BS(F, K, T) under SABR.
    """
    F = _ensure_numpy(F)
    K = _ensure_numpy(K)
    T = _ensure_numpy(T)

    # Broadcast to common shape
    F, K, T = np.broadcast_arrays(F, K, T)

    # Input sanitization (domain constraints, not tolerances)
    alpha = max(float(alpha), Tolerance.LOG_MIN)
    beta = float(np.clip(beta, 0.0, 1.0))
    rho = float(np.clip(rho, -_RHO_BOUND, _RHO_BOUND))
    nu = max(float(nu), Tolerance.LOG_MIN)
    shift = float(shift)

    # Apply shift (shifted-SABR)
    Fs = np.maximum(F + shift, Tolerance.LOG_MIN)
    Ks = np.maximum(K + shift, Tolerance.LOG_MIN)

    one_minus_beta = 1.0 - beta
    FK_pow = (Fs * Ks) ** (0.5 * one_minus_beta)
    logFK = np.log(Fs / Ks)

    # z and chi(z). Standard Hagan: z = (nu/alpha) * (F K)^((1-beta)/2) * ln(F/K).
    # (The legacy implementation carried a spurious 1/(1-beta) factor that both
    # mis-scaled the smile and divided by zero at beta=1; removed here.)
    z = (nu / alpha) * FK_pow * logFK
    chi = _chi_z(z, rho)

    # x = z / chi(z) stabilized: x -> 1 as z -> 0
    x = np.ones_like(z)
    mask = np.abs(z) > _X_LIMIT_RADIUS
    x[mask] = z[mask] / chi[mask]

    # A(T) time-dependent correction term.
    FK_pow_1mb = (Fs * Ks) ** one_minus_beta
    FK_pow_half_1mb = (Fs * Ks) ** (0.5 * one_minus_beta)

    term1 = ((one_minus_beta**2) * (alpha**2)) / (
        24.0 * np.maximum(FK_pow_1mb, Tolerance.LOG_MIN)
    )
    term2 = (rho * beta * nu * alpha) / (
        4.0 * np.maximum(FK_pow_half_1mb, Tolerance.LOG_MIN)
    )
    term3 = (2.0 - 3.0 * rho * rho) * (nu * nu) / 24.0
    A = 1.0 + (term1 + term2 + term3) * T

    # Leading factor. Hagan's denominator includes a log-moneyness correction
    # D = (F K)^((1-beta)/2) * (1 + (1-beta)^2/24 ln^2(F/K) + (1-beta)^4/1920 ln^4(F/K)).
    log2 = logFK * logFK
    denom_corr = (
        1.0
        + (one_minus_beta**2) * log2 / 24.0
        + (one_minus_beta**4) * (log2 * log2) / 1920.0
    )
    leading = alpha / np.maximum(FK_pow * denom_corr, Tolerance.LOG_MIN)

    sigma = np.asarray(leading * A * x)

    # Handle exact ATM Ks == Fs using the ATM specialization (better accuracy).
    # ``np.where`` keeps this correct for scalar/0-d inputs (where boolean-mask
    # item assignment would fail on a collapsed numpy scalar).
    atm_mask = np.abs(Fs - Ks) < _ATM_GAP
    if np.any(atm_mask):
        sigma_atm_vals = sabr_atm_implied_vol_black(Fs, T, alpha, beta, rho, nu)
        sigma = np.where(atm_mask, sigma_atm_vals, sigma)

    return sigma


def sabr_atm_implied_vol_black(
    F: ArrayLike,
    T: ArrayLike,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
) -> np.ndarray:
    """
    Hagan's ATM-forward Black implied volatility approximation.

    For K = F, the general-strike formula reduces to this simpler form.

    Parameters
    ----------
    F : float or array
        Forward.
    T : float or array
        Time to maturity in years.
    alpha, beta, rho, nu, shift : see `sabr_implied_vol_black`.

    Returns
    -------
    np.ndarray
        ATM Black implied vol.
    """
    F = _ensure_numpy(F)
    T = _ensure_numpy(T)
    F, T = np.broadcast_arrays(F, T)

    alpha = max(float(alpha), Tolerance.LOG_MIN)
    beta = float(np.clip(beta, 0.0, 1.0))
    rho = float(np.clip(rho, -_RHO_BOUND, _RHO_BOUND))
    nu = max(float(nu), Tolerance.LOG_MIN)
    shift = float(shift)

    Fs = np.maximum(F + shift, Tolerance.LOG_MIN)
    one_minus_beta = 1.0 - beta

    # ATM specialization (z -> 0, x -> 1)
    base = alpha / np.power(Fs, one_minus_beta)

    term1 = ((one_minus_beta**2) * (alpha**2)) / (
        24.0 * np.power(Fs, 2.0 * one_minus_beta)
    )
    term2 = (rho * beta * nu * alpha) / (4.0 * np.power(Fs, one_minus_beta))
    term3 = (2.0 - 3.0 * rho * rho) * (nu * nu) / 24.0
    A = 1.0 + (term1 + term2 + term3) * T

    return base * A


def sabr_implied_vol_black_shifted(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    m: float,
) -> np.ndarray:
    """
    Convenience wrapper for shifted SABR (alias of sabr_implied_vol_black with
    shift=m).
    """
    return sabr_implied_vol_black(F, K, T, alpha, beta, rho, nu, shift=float(m))


def sabr_generate_vol_surface(
    F: float,
    strikes: Iterable[float],
    maturities: Iterable[float],
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
) -> dict[tuple[float, float], float]:
    """
    Generate a SABR Black implied volatility surface on a strike-maturity grid.

    Returns a dict keyed by (K, T) with vol values.
    """
    strikes_arr = _ensure_numpy(np.array(list(strikes), dtype=float))
    maturities_arr = _ensure_numpy(np.array(list(maturities), dtype=float))

    surface: dict[tuple[float, float], float] = {}
    for T in maturities_arr:
        vols = sabr_implied_vol_black(
            F, strikes_arr, T * np.ones_like(strikes_arr), alpha, beta, rho, nu, shift
        )
        for K, v in zip(strikes_arr, vols):
            surface[(float(K), float(T))] = float(v)
    return surface


__all__ = [
    "sabr_implied_vol_black",
    "sabr_implied_vol_black_shifted",
    "sabr_atm_implied_vol_black",
    "sabr_generate_vol_surface",
]
