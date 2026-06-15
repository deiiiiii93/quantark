#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SABR calibration utilities.

Simple, dependency-light calibrators for SABR parameters using Hagan's
lognormal implied volatility approximation.

Design goals:
- Avoid a SciPy optimizer dependency via coarse-to-fine grid search plus a
  stabilized Newton/secant step for alpha.
- Robust defaults and bounds.
- Clear, maintainable code.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance

from .hagan import sabr_atm_implied_vol_black, sabr_implied_vol_black


def _as_np(x) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else np.asarray(x, dtype=float)


def _weighted_mse(
    y_true: np.ndarray, y_pred: np.ndarray, w: Optional[np.ndarray]
) -> float:
    if w is None:
        return float(np.mean((y_true - y_pred) ** 2))
    w_arr = _as_np(w)
    w_norm = w_arr / np.maximum(np.sum(w_arr), Tolerance.LOG_MIN)
    return float(np.sum(w_norm * (y_true - y_pred) ** 2))


def _solve_alpha_from_atm(
    F: float,
    T: float,
    sigma_atm: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
    max_iter: int = 25,
) -> float:
    """
    Solve for alpha using the SABR ATM-forward formula via a stabilized Newton
    method.

    Initial guess alpha0 = sigma_atm * (F + shift)^{1-beta}, then refined with a
    central-difference numerical derivative and damping.
    """
    F_eff = max(F + shift, Tolerance.LOG_MIN)
    beta = float(np.clip(beta, 0.0, 1.0))
    sigma_atm = max(float(sigma_atm), Tolerance.PRECISION)

    alpha = sigma_atm * (F_eff ** (1.0 - beta))
    for _ in range(max_iter):
        f_val = (
            sabr_atm_implied_vol_black(F, T, alpha, beta, rho, nu, shift) - sigma_atm
        )
        if abs(float(f_val)) < Tolerance.LOG_MIN:
            break
        # Numerical derivative d f / d alpha (central difference)
        step = Tolerance.PRECISION * max(1.0, abs(alpha))
        v_up = sabr_atm_implied_vol_black(F, T, alpha + step, beta, rho, nu, shift)
        v_dn = sabr_atm_implied_vol_black(F, T, alpha - step, beta, rho, nu, shift)
        deriv = (float(v_up) - float(v_dn)) / (2.0 * step)
        if abs(deriv) < Tolerance.LOG_MIN:
            break
        # Damped Newton step (keep alpha positive)
        delta = f_val / deriv
        alpha = max(alpha - 0.75 * float(delta), Tolerance.PRECISION)

    return float(alpha)


def calibrate_sabr_slice(
    F: float,
    strikes: Iterable[float],
    T: float,
    market_vols: Iterable[float],
    beta: float = 0.5,
    shift: float = 0.0,
    weights: Optional[Iterable[float]] = None,
    alpha_bounds: Tuple[float, float] = (1e-4, 5.0),
    nu_bounds: Tuple[float, float] = (1e-4, 5.0),
    rho_bounds: Tuple[float, float] = (-0.999, 0.999),
    grid_size: int = 41,
    refine: bool = True,
) -> Dict[str, float]:
    """
    Calibrate SABR parameters (alpha, rho, nu) for a single maturity smile.

    Strategy:
      1) Solve alpha from the ATM vol (closest strike to forward).
      2) Grid search over (rho, nu) to minimize weighted MSE of implied vols.
      3) Optional local refinement on a denser grid around the best point.

    Raises:
        ValidationError: If strikes and market_vols differ in length, or all
            market vols are non-finite.
    """
    strikes = _as_np(np.array(list(strikes), dtype=float))
    market_vols = _as_np(np.array(list(market_vols), dtype=float))
    T = float(T)

    if strikes.shape != market_vols.shape:
        raise ValidationError("strikes and market_vols must have the same length")

    weights_arr = None if weights is None else _as_np(np.asarray(list(weights), dtype=float))
    if weights_arr is not None and weights_arr.shape != market_vols.shape:
        raise ValidationError("weights must have the same length as market_vols")

    # Drop non-finite quotes (and their strikes/weights) before fitting: a single
    # NaN/inf would otherwise poison every objective evaluation with NaN and the
    # search would silently return its defaults.
    finite_mask = np.isfinite(market_vols)
    if not np.any(finite_mask):
        raise ValidationError("market_vols contains no finite values to calibrate to")
    strikes = strikes[finite_mask]
    market_vols = market_vols[finite_mask]
    if weights_arr is not None:
        weights_arr = weights_arr[finite_mask]

    # Identify ATM-like strike (closest to forward)
    idx_atm = int(np.argmin(np.abs(strikes - F)))
    sigma_atm = float(market_vols[idx_atm])

    beta = float(np.clip(beta, 0.0, 1.0))

    # Step 1: alpha from ATM. If sigma_atm is degenerate, use the median of the
    # (now finite) market vols (data-robustness branch, not a model fallback).
    if not np.isfinite(sigma_atm) or sigma_atm <= 0.0:
        sigma_atm = float(np.median(market_vols))

    # Stabilize alpha across a few provisional (rho, nu) pairs; pick the median.
    provisional_grid = [(-0.3, 0.5), (0.0, 0.5), (-0.5, 1.0), (0.3, 1.0)]
    alphas = []
    for prho, pnu in provisional_grid:
        a = _solve_alpha_from_atm(F, T, sigma_atm, beta, prho, pnu, shift)
        if alpha_bounds[0] <= a <= alpha_bounds[1]:
            alphas.append(a)
    alpha_guess = (
        float(np.median(alphas))
        if alphas
        else _solve_alpha_from_atm(F, T, sigma_atm, beta, 0.0, 0.5, shift)
    )
    alpha_guess = float(np.clip(alpha_guess, alpha_bounds[0], alpha_bounds[1]))

    # Step 2: Grid search for (rho, nu)
    def objective(alpha: float, rho: float, nu: float) -> float:
        model_vols = sabr_implied_vol_black(
            F,
            strikes,
            T * np.ones_like(strikes),
            alpha,
            beta,
            rho,
            nu,
            shift=shift,
        )
        return _weighted_mse(market_vols, model_vols, weights_arr)

    rho_grid = np.linspace(rho_bounds[0], rho_bounds[1], grid_size)
    nu_grid = np.linspace(nu_bounds[0], nu_bounds[1], grid_size)

    best_err = float("inf")
    best_rho, best_nu = 0.0, 0.5
    for rho in rho_grid:
        for nu in nu_grid:
            err = objective(alpha_guess, float(rho), float(nu))
            if err < best_err:
                best_err = err
                best_rho, best_nu = float(rho), float(nu)

    if refine:
        # Local refinement with a smaller grid around the best point.
        rho_lo = max(rho_bounds[0], best_rho - (rho_grid[1] - rho_grid[0]) * 3)
        rho_hi = min(rho_bounds[1], best_rho + (rho_grid[1] - rho_grid[0]) * 3)
        nu_lo = max(nu_bounds[0], best_nu - (nu_grid[1] - nu_grid[0]) * 3)
        nu_hi = min(nu_bounds[1], best_nu + (nu_grid[1] - nu_grid[0]) * 3)

        rho_grid_ref = np.linspace(rho_lo, rho_hi, 21)
        nu_grid_ref = np.linspace(nu_lo, nu_hi, 21)
        for rho in rho_grid_ref:
            for nu in nu_grid_ref:
                err = objective(alpha_guess, float(rho), float(nu))
                if err < best_err:
                    best_err = err
                    best_rho, best_nu = float(rho), float(nu)

    # Micro re-solve of alpha given the best (rho, nu), then re-evaluate the fit
    # so the returned mse is consistent with the returned (alpha, rho, nu).
    alpha_star = _solve_alpha_from_atm(F, T, sigma_atm, beta, best_rho, best_nu, shift)
    alpha_star = float(np.clip(alpha_star, alpha_bounds[0], alpha_bounds[1]))
    final_mse = objective(alpha_star, best_rho, best_nu)

    return {
        "alpha": alpha_star,
        "beta": beta,
        "rho": best_rho,
        "nu": best_nu,
        "shift": shift,
        "mse": final_mse,
    }


def calibrate_sabr_surface(
    F: float,
    strikes: Iterable[float],
    maturities: Iterable[float],
    market_vol_matrix: np.ndarray,
    beta: float = 0.5,
    shift: float = 0.0,
    weights: Optional[np.ndarray] = None,
    alpha_bounds: Tuple[float, float] = (1e-4, 5.0),
    nu_bounds: Tuple[float, float] = (1e-4, 5.0),
    rho_bounds: Tuple[float, float] = (-0.999, 0.999),
    grid_size: int = 41,
    refine: bool = True,
) -> Dict[float, Dict[str, float]]:
    """
    Calibrate SABR parameters per maturity slice. Returns a dict keyed by T.

    Parameters
    ----------
    F : float
        Forward at t0 (assumed common across maturities for the equity/FX
        forward approximation).
    strikes : Iterable[float]
        Strike grid.
    maturities : Iterable[float]
        Maturities in years, length M.
    market_vol_matrix : (M, N) array
        Market Black implied vols per maturity-strike.
    beta, shift : float
        SABR beta (fixed) and shift.
    weights : Optional[(M, N) array]
        Optional weights per data point.

    Raises:
        ValidationError: If matrix/weights shapes are inconsistent.
    """
    strikes = _as_np(np.array(list(strikes), dtype=float))
    maturities = _as_np(np.array(list(maturities), dtype=float))
    vols = _as_np(market_vol_matrix)

    if vols.shape != (len(maturities), len(strikes)):
        raise ValidationError(
            "market_vol_matrix shape must be (len(maturities), len(strikes))"
        )

    if weights is not None:
        weights = _as_np(weights)
        if weights.shape != vols.shape:
            raise ValidationError("weights shape must match market_vol_matrix")

    results: Dict[float, Dict[str, float]] = {}
    for i, T in enumerate(maturities):
        w_slice = None if weights is None else weights[i]
        res = calibrate_sabr_slice(
            F=F,
            strikes=strikes,
            T=float(T),
            market_vols=vols[i],
            beta=beta,
            shift=shift,
            weights=w_slice,
            alpha_bounds=alpha_bounds,
            nu_bounds=nu_bounds,
            rho_bounds=rho_bounds,
            grid_size=grid_size,
            refine=refine,
        )
        results[float(T)] = res

    return results


__all__ = [
    "calibrate_sabr_slice",
    "calibrate_sabr_surface",
]
