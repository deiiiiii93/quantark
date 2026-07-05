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
        # Arithmetic Euler can overshoot below zero; apply an absorbing boundary at S=0
        # (a price at zero stays at zero). EULERLOG/QUADEXP preserve positivity exactly
        # and are the recommended schemes.
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

    if scheme in (HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M):
        martingale = scheme == HestonMCScheme.QUADEXP_M
        psi_c = 1.5
        deterministic_vol = sigma <= 1e-8
        # When variance is deterministic, spot diffusion is the FULL Brownian (no
        # correlation to a non-existent variance shock); otherwise the rho-correlated
        # part is reconstructed via corr and the independent part scales by rho_hat.
        diff_coef = 1.0 if deterministic_vol else rho_hat
        log_s = np.full(n_paths, np.log(max(float(s0), 1e-12)))
        v_n = np.full(n_paths, max(float(v0), 0.0))
        for i in range(M):
            dt = step_dt[i]
            sqrt_dt = np.sqrt(dt)
            drift = r_fwd[i] - carry_fwd[i]
            exp_kdt = np.exp(-kappa * dt)
            omexp = -np.expm1(-kappa * dt)  # 1 - e^{-k dt}, stable
            m = theta + (v_n - theta) * exp_kdt
            if kappa > _KMIN:
                inv_k = 1.0 / kappa
                s2 = (
                    v_n * sigma2 * exp_kdt * (omexp * inv_k)
                    + theta * sigma2 * (omexp * omexp * inv_k) / 2.0
                )
            else:
                # kappa -> 0 limit: (1-e^{-k dt})/k -> dt, second term -> 0.
                s2 = v_n * sigma2 * dt
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

            # Branch B: exponential with Bernoulli mass at zero (psi > psi_c).
            # Inverse CDF (Andersen 2008, eq. 25): Psi^{-1}(u) = ln((1-p)/(1-u)) / beta,
            # POSITIVE for u > p. A historical sign error negated it, so every branch-B
            # draw was clamped to 0 and the variance collapsed whenever psi > psi_c
            # (Feller-violated regimes), leaving a persistent martingale bias in spot.
            p = np.clip((psi - 1.0) / (psi + 1.0), 0.0, 0.999999)
            beta = np.maximum((1.0 - p) / np.maximum(m, _KMIN), _KMIN)
            u_clip = np.clip(u_var[:, i], 1e-12, 1.0 - 1e-12)
            with np.errstate(divide="ignore", invalid="ignore"):
                v_b = np.where(u_clip <= p, 0.0, np.log((1.0 - p) / (1.0 - u_clip)) / beta)

            v_np = np.where(psi <= psi_c, v_a, v_b)
            v_np = np.maximum(v_np, 0.0)

            v_bar = np.maximum(0.5 * (v_np + np.maximum(v_n, 0.0)), 0.0)
            if deterministic_vol:
                corr = 0.0
            else:
                corr = (rho / sigma) * (v_np - v_n - kappa * (theta - v_bar) * dt)
            if martingale and not deterministic_vol:
                # Andersen §4.2 Prop. 4.1: swap the approximate constant K0 = -rho*kappa*theta*dt/sigma
                # for the exact per-path K0* so E[S_{t+dt}|F_t] = S_t*e^{drift*dt} exactly.
                ros = rho / sigma
                K3 = 0.5 * (1.0 - rho * rho) * dt          # == K4 (central gamma = 1/2)
                K1 = 0.5 * dt * (kappa * ros - 0.5) - ros
                K2 = 0.5 * dt * (kappa * ros - 0.5) + ros
                A = K2 + 0.5 * K3                           # coefficient on V_{t+dt} after E_Z
                quad_mask = psi <= psi_c
                denom_q = 1.0 - 2.0 * A * a                 # quadratic-branch MGF domain
                denom_e = beta - A                          # exponential-branch MGF domain
                bad = (quad_mask & (denom_q <= 0.0)) | (~quad_mask & (denom_e <= 0.0))
                if np.any(bad):
                    from quantark.util.exceptions import NumericalError
                    raise NumericalError(
                        "QE-M martingale MGF is undefined at these parameters "
                        "(A outside the CIR-transition MGF domain); tighten dt or use QUADEXP"
                    )
                safe_q = np.where(denom_q > 0.0, denom_q, 1.0)
                safe_e = np.where(denom_e > 0.0, denom_e, 1.0)
                m_quad = np.exp(A * a * b * b / safe_q) / np.sqrt(safe_q)
                m_exp = p + (1.0 - p) * beta / safe_e
                M = np.where(quad_mask, m_quad, m_exp)
                ln_M = np.log(M)
                K0 = -ros * kappa * theta * dt
                K0_star = -ln_M - (K1 + 0.5 * K3) * v_n
                # replace K0 with K0* on top of the standard increment
                log_s = (log_s + (drift - 0.5 * v_bar) * dt + corr - K0 + K0_star
                         + np.sqrt(v_bar) * sqrt_dt * diff_coef * z_ind[:, i])
            else:
                # QUADEXP: preserve the exact original expression grouping (bit-identical seed pins).
                log_s = log_s + (drift - 0.5 * v_bar) * dt + corr + np.sqrt(v_bar) * sqrt_dt * diff_coef * z_ind[:, i]
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
    sampler=None,
    return_stderr: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Price a European vanilla under Heston via Monte Carlo (terminal spot).

    sampler (optional): a quantark.montecarlo generator exposing ``uniform(n, dim)``.
        When provided, draws one low-discrepancy uniform block of dimension
        ``n_streams*M`` and splits columns [z_var(M) | z_ind(M) | u_var(M)] (the u block
        present only for QE/QE-M), transforming the z columns via ndtri. Mutually
        exclusive with ``use_antithetic``. Default None keeps the pseudo path bit-identical.
    """
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

    half = (num_paths + 1) // 2
    n_eff = 2 * half if use_antithetic else num_paths

    # Uniforms are consumed only by the QE variance inverse-CDF; EULER/EULERLOG skip the
    # draw entirely. u draws come after the z draws, so the z-streams (and QUADEXP's
    # u-stream) are seed-identical to the always-draw layout.
    need_u = scheme in (HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M)
    if sampler is not None:
        if use_antithetic:
            raise ValidationError("sampler and use_antithetic are mutually exclusive")
        from scipy.special import ndtri
        n_streams = 3 if need_u else 2          # [z_var | z_ind | (u_var)]
        block = np.asarray(sampler.uniform(num_paths, n_streams * M), dtype=float)
        block = np.clip(block, 1e-12, 1.0 - 1e-12)
        z_var = ndtri(block[:, 0:M])
        z_ind = ndtri(block[:, M:2 * M])
        u_var = block[:, 2 * M:3 * M] if need_u else None
    else:
        rng = np.random.default_rng(seed)
        if use_antithetic:
            z_var_h = rng.standard_normal((half, M))
            z_ind_h = rng.standard_normal((half, M))
            z_var = np.concatenate([z_var_h, -z_var_h], axis=0)
            z_ind = np.concatenate([z_ind_h, -z_ind_h], axis=0)
            if need_u:
                u_var_h = rng.random((half, M))
                u_var = np.concatenate([u_var_h, 1.0 - u_var_h], axis=0)
            else:
                u_var = None
        else:
            z_var = rng.standard_normal((n_eff, M))
            z_ind = rng.standard_normal((n_eff, M))
            u_var = rng.random((n_eff, M)) if need_u else None

    s_terminal = _simulate_terminal_spot(s0, params, scheme, dt, rf, cf, z_var, z_ind, u_var)
    if not np.all(np.isfinite(s_terminal)):
        from quantark.util.exceptions import NumericalError
        raise NumericalError("Heston MC produced non-finite terminal spots (extreme parameters)")
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
