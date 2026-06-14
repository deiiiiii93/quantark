"""Heston Monte Carlo terminal-price kernel (vectorized; no numba).

Three time-discretization schemes ported from the SLV reference:
- EULER:    full-truncation Euler with Milstein correction on arithmetic spot.
- EULERLOG: full-truncation Euler in log-spot.
- QUADEXP:  Andersen (2008) Quadratic-Exponential variance scheme with the
            martingale-correcting drift (correlation term).

Rates enter as per-step forwards (drift_i = r_fwd[i] - carry_fwd[i]); the variance
process is rate-independent. Asset-neutral: carry = dividend yield (equity) or foreign
rate (FX). Returns the discounted European price (terminal spot only).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston.params import HestonParams

_KMIN = 1e-12


def _simulate_terminal_spot(
    s0, params, scheme, step_dt, r_fwd, carry_fwd, z_var, z_ind, u_var,
):
    """Evolve (log) spot and variance to maturity for all paths; return terminal spot."""
    kappa, theta, sigma, rho, v0 = (
        params.kappa, params.theta, params.sigma, params.rho, params.v0,
    )
    rho_hat = np.sqrt(max(1.0 - rho * rho, 0.0))
    sigma2 = sigma * sigma
    n_paths = z_var.shape[0]
    M = step_dt.size

    if scheme == HestonMCScheme.EULER:
        s = np.full(n_paths, float(s0))
        v = np.full(n_paths, float(v0))
        for i in range(M):
            dt = step_dt[i]
            sqrt_dt = np.sqrt(dt)
            drift = r_fwd[i] - carry_fwd[i]
            z1 = z_var[:, i] * sqrt_dt
            z2 = z_ind[:, i] * sqrt_dt
            z_s = rho * z1 + rho_hat * z2
            v_plus = np.maximum(v, 0.0)
            sqrt_vp = np.sqrt(v_plus)
            v = v + kappa * (theta - v_plus) * dt + sigma * sqrt_vp * z1 + 0.25 * sigma2 * (z1 * z1 - dt)
            s = s + drift * s * dt + sqrt_vp * s * z_s + 0.5 * s * v_plus * (z_s * z_s - dt)
        return np.maximum(s, 0.0)

    if scheme == HestonMCScheme.EULERLOG:
        log_s = np.full(n_paths, np.log(s0))
        v = np.full(n_paths, float(v0))
        for i in range(M):
            dt = step_dt[i]
            sqrt_dt = np.sqrt(dt)
            drift = r_fwd[i] - carry_fwd[i]
            z1 = z_var[:, i] * sqrt_dt
            z_s = rho * z1 + rho_hat * z_ind[:, i] * sqrt_dt
            v_plus = np.maximum(v, 0.0)
            sqrt_vp = np.sqrt(v_plus)
            log_s = log_s + (drift - 0.5 * v_plus) * dt + sqrt_vp * z_s
            v = v + kappa * (theta - v_plus) * dt + sigma * sqrt_vp * z1 + 0.25 * sigma2 * (z1 * z1 - dt)
        return np.exp(log_s)

    if scheme == HestonMCScheme.QUADEXP:
        psi_c = 1.5
        log_s = np.full(n_paths, np.log(max(float(s0), 1e-12)))
        v_n = np.full(n_paths, max(float(v0), 0.0))
        for i in range(M):
            dt = step_dt[i]
            sqrt_dt = np.sqrt(dt)
            drift = r_fwd[i] - carry_fwd[i]
            exp_kdt = np.exp(-kappa * dt)
            k_safe = max(kappa, _KMIN)
            m = theta + (v_n - theta) * exp_kdt
            s2 = (
                v_n * sigma2 * exp_kdt * (1.0 - exp_kdt) / k_safe
                + theta * sigma2 * (1.0 - exp_kdt) ** 2 / (2.0 * k_safe)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                psi = np.where(m <= 1e-12, 0.0, s2 / (m * m))
            psi = np.maximum(psi, 0.0)

            # Branch A: quadratic (psi <= psi_c)
            phi = 2.0 / np.maximum(psi, 1e-16)
            rad = np.maximum(phi * (phi - 1.0), 0.0)
            B = np.maximum(phi - 1.0 + np.sqrt(rad), 0.0)
            b = np.sqrt(B)
            a = m / (1.0 + b * b)
            zv = z_var[:, i]
            v_a = a * (b + zv) * (b + zv)

            # Branch B: exponential with Bernoulli mass at zero (psi > psi_c)
            p = np.clip((psi - 1.0) / (psi + 1.0), 0.0, 0.999999)
            beta = np.maximum((1.0 - p) / np.maximum(m, _KMIN), _KMIN)
            u_clip = np.clip(u_var[:, i], 1e-12, 1.0 - 1e-12)
            with np.errstate(divide="ignore", invalid="ignore"):
                v_b = np.where(u_clip <= p, 0.0, -np.log((1.0 - p) / (1.0 - u_clip)) / beta)

            v_np = np.where(psi <= psi_c, v_a, v_b)
            v_np = np.maximum(v_np, 0.0)

            v_bar = np.maximum(0.5 * (v_np + np.maximum(v_n, 0.0)), 0.0)
            if sigma <= 1e-8:
                corr = 0.0
            else:
                corr = (rho / sigma) * (v_np - v_n - kappa * (theta - v_bar) * dt)
            log_s = log_s + (drift - 0.5 * v_bar) * dt + corr + np.sqrt(v_bar) * sqrt_dt * rho_hat * z_ind[:, i]
            v_n = v_np
        return np.exp(log_s)

    raise ValidationError(f"unknown Heston MC scheme: {scheme}")


def price_european_heston_mc(
    s0: float,
    strike: float,
    is_call: bool,
    params: HestonParams,
    step_dt: np.ndarray,
    r_fwd: np.ndarray,
    carry_fwd: np.ndarray,
    disc_factor: float,
    scheme: HestonMCScheme = HestonMCScheme.QUADEXP,
    num_paths: int = 50_000,
    seed: Optional[int] = 42,
    use_antithetic: bool = False,
    return_stderr: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Price a European vanilla under Heston via Monte Carlo (terminal spot)."""
    dt = np.asarray(step_dt, dtype=float)
    rf = np.asarray(r_fwd, dtype=float)
    cf = np.asarray(carry_fwd, dtype=float)
    M = dt.size
    if M < 1 or rf.size != M or cf.size != M:
        raise ValidationError("step_dt, r_fwd, carry_fwd must be equal-length, length >= 1")
    if not (np.all(np.isfinite(dt)) and np.all(dt > 0)):
        raise ValidationError("step_dt must be finite and positive")
    if not (np.all(np.isfinite(rf)) and np.all(np.isfinite(cf))):
        raise ValidationError("r_fwd and carry_fwd must be finite")
    if s0 <= 0 or strike <= 0:
        raise ValidationError("s0 and strike must be positive")
    if not np.isfinite(disc_factor) or disc_factor <= 0:
        raise ValidationError("disc_factor must be finite and positive")
    if num_paths <= 0:
        raise ValidationError("num_paths must be positive")
    if not isinstance(scheme, HestonMCScheme):
        raise ValidationError("scheme must be a HestonMCScheme")

    rng = np.random.default_rng(seed)
    half = (num_paths + 1) // 2
    n_eff = 2 * half if use_antithetic else num_paths

    if use_antithetic:
        z_var_h = rng.standard_normal((half, M))
        z_ind_h = rng.standard_normal((half, M))
        u_var_h = rng.random((half, M))
        z_var = np.concatenate([z_var_h, -z_var_h], axis=0)
        z_ind = np.concatenate([z_ind_h, -z_ind_h], axis=0)
        u_var = np.concatenate([u_var_h, 1.0 - u_var_h], axis=0)
    else:
        z_var = rng.standard_normal((n_eff, M))
        z_ind = rng.standard_normal((n_eff, M))
        u_var = rng.random((n_eff, M))

    s_terminal = _simulate_terminal_spot(s0, params, scheme, dt, rf, cf, z_var, z_ind, u_var)
    payoff = np.maximum(s_terminal - strike, 0.0) if is_call else np.maximum(strike - s_terminal, 0.0)
    discounted = float(disc_factor) * payoff

    if use_antithetic:
        pair = 0.5 * (discounted[:half] + discounted[half:2 * half])
        price = float(np.mean(pair))
        if return_stderr:
            return price, (float(np.std(pair, ddof=1) / np.sqrt(half)) if half > 1 else 0.0)
        return price

    price = float(np.mean(discounted))
    if return_stderr:
        return price, (float(np.std(discounted, ddof=1) / np.sqrt(num_paths)) if num_paths > 1 else 0.0)
    return price
