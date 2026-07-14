"""Heston Monte Carlo terminal-price kernel (vectorized; no numba).

Heston time-discretization schemes ported from the SLV reference:
- EULER:    full-truncation Euler with Milstein correction on arithmetic spot.
- EULERLOG: full-truncation Euler in log-spot.
- FULL_TRUNCATION_EULER: log-spot with the plain full-truncation variance update
            historically used by the DCN engine (no Milstein correction).
- QUADEXP:  Andersen (2008) Quadratic-Exponential variance scheme with the
            martingale-correcting drift (correlation term).
- QUADEXP_M: QUADEXP with Andersen's exact conditional martingale correction.

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


def _simulate_heston_spots(
    s0,
    params,
    scheme,
    step_dt,
    r_fwd,
    carry_fwd,
    z_var,
    z_ind,
    u_var=None,
    record_steps=None,
    _terminal_only=False,
):
    """Shared Heston evolution core for terminal-only and recorded-node paths.

    Random draws are supplied by the caller so product engines can share one
    canonical stream layout. ``record_steps`` uses node indices in ``[0, M]``;
    omitting it records every node. This keeps path-dependent engines from
    allocating unneeded fine-grid nodes when they use simulation substeps.
    """
    if not isinstance(params, HestonParams):
        raise ValidationError("params must be a HestonParams")
    if not isinstance(scheme, HestonMCScheme):
        raise ValidationError("scheme must be a HestonMCScheme")
    if not np.isfinite(s0) or s0 <= 0.0:
        raise ValidationError("s0 must be finite and positive")

    step_dt = np.asarray(step_dt, dtype=float)
    r_fwd = np.asarray(r_fwd, dtype=float)
    carry_fwd = np.asarray(carry_fwd, dtype=float)
    z_var = np.asarray(z_var, dtype=float)
    z_ind = np.asarray(z_ind, dtype=float)
    if step_dt.ndim != 1 or step_dt.size < 1:
        raise ValidationError("step_dt must be one-dimensional and non-empty")
    if r_fwd.shape != step_dt.shape or carry_fwd.shape != step_dt.shape:
        raise ValidationError("step_dt, r_fwd, carry_fwd must have equal shapes")
    if not (np.all(np.isfinite(step_dt)) and np.all(step_dt > 0.0)):
        raise ValidationError("step_dt must be finite and positive")
    if not (np.all(np.isfinite(r_fwd)) and np.all(np.isfinite(carry_fwd))):
        raise ValidationError("r_fwd and carry_fwd must be finite")
    if z_var.ndim != 2 or z_var.shape != z_ind.shape:
        raise ValidationError("z_var and z_ind must be equal-shape 2D arrays")

    n_paths, n_steps = z_var.shape
    if n_paths < 1 or n_steps != step_dt.size:
        raise ValidationError(
            "random draw arrays must have shape (n_paths, len(step_dt))"
        )
    needs_uniform = scheme in (
        HestonMCScheme.QUADEXP,
        HestonMCScheme.QUADEXP_M,
    )
    if needs_uniform:
        if u_var is None or np.asarray(u_var).shape != z_var.shape:
            raise ValidationError(
                "u_var must match z_var for QUADEXP/QUADEXP_M"
            )
        u_var = np.asarray(u_var, dtype=float)

    if _terminal_only:
        nodes = None
        record_col = None
    else:
        if record_steps is None:
            records = np.arange(n_steps + 1, dtype=int)
        else:
            raw_records = np.asarray(record_steps)
            if raw_records.ndim != 1 or raw_records.size < 1:
                raise ValidationError("record_steps must be a non-empty 1D array")
            if not np.issubdtype(raw_records.dtype, np.integer):
                raise ValidationError("record_steps must contain integers")
            records = raw_records.astype(int, copy=False)
            if (
                records[0] < 0
                or records[-1] > n_steps
                or np.any(np.diff(records) <= 0)
            ):
                raise ValidationError(
                    "record_steps must be strictly increasing and lie in [0, M]"
                )

        nodes = np.empty((n_paths, records.size), dtype=float)
        record_col = np.full(n_steps + 1, -1, dtype=int)
        record_col[records] = np.arange(records.size)

    def _record(step, spot):
        if record_col is None:
            return
        col = int(record_col[step])
        if col >= 0:
            nodes[:, col] = spot

    def _record_log(step, log_spot):
        if record_col is None:
            return
        col = int(record_col[step])
        if col >= 0:
            nodes[:, col] = np.exp(log_spot)

    def _record_nonnegative(step, spot):
        if record_col is None:
            return
        col = int(record_col[step])
        if col >= 0:
            nodes[:, col] = np.maximum(spot, 0.0)

    kappa, theta, sigma, rho, v0 = (
        params.kappa, params.theta, params.sigma, params.rho, params.v0,
    )
    rho_hat = np.sqrt(max(1.0 - rho * rho, 0.0))
    sigma2 = sigma * sigma

    if scheme == HestonMCScheme.EULER:
        s = np.full(n_paths, float(s0))
        v = np.full(n_paths, float(v0))
        _record(0, s)
        for i in range(n_steps):
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
            _record_nonnegative(i + 1, s)
        # Arithmetic Euler can overshoot below zero; apply an absorbing boundary at S=0
        # (a price at zero stays at zero). EULERLOG/QUADEXP preserve positivity exactly
        # and are the recommended schemes.
        return np.maximum(s, 0.0) if _terminal_only else nodes

    if scheme == HestonMCScheme.FULL_TRUNCATION_EULER:
        # This branch intentionally preserves the original DCN update, including
        # its endpoint correlation clipping and absence of a variance Milstein term.
        legacy_rho = float(np.clip(rho, -0.999, 0.999))
        legacy_rho_hat = np.sqrt(max(1.0 - legacy_rho * legacy_rho, 0.0))
        log_s = np.full(n_paths, np.log(float(s0)))
        v = np.full(n_paths, max(float(v0), 0.0))
        _record_log(0, log_s)
        for i in range(n_steps):
            dt = step_dt[i]
            sqrt_dt = np.sqrt(dt)
            drift = r_fwd[i] - carry_fwd[i]
            v_plus = np.maximum(v, 0.0)
            sqrt_vp = np.sqrt(v_plus)
            d_w_v = z_var[:, i] * sqrt_dt
            d_w_s = (
                legacy_rho * z_var[:, i] + legacy_rho_hat * z_ind[:, i]
            ) * sqrt_dt
            log_s = (
                log_s
                + (drift - 0.5 * v_plus) * dt
                + sqrt_vp * d_w_s
            )
            v = (
                v
                + kappa * (theta - v_plus) * dt
                + sigma * sqrt_vp * d_w_v
            )
            _record_log(i + 1, log_s)
        return np.exp(log_s) if _terminal_only else nodes

    if scheme == HestonMCScheme.EULERLOG:
        log_s = np.full(n_paths, np.log(s0))
        v = np.full(n_paths, float(v0))
        _record_log(0, log_s)
        for i in range(n_steps):
            dt = step_dt[i]
            sqrt_dt = np.sqrt(dt)
            drift = r_fwd[i] - carry_fwd[i]
            z1 = z_var[:, i] * sqrt_dt
            z_s = rho * z1 + rho_hat * z_ind[:, i] * sqrt_dt
            v_plus = np.maximum(v, 0.0)
            sqrt_vp = np.sqrt(v_plus)
            log_s = log_s + (drift - 0.5 * v_plus) * dt + sqrt_vp * z_s
            v = v + kappa * (theta - v_plus) * dt + sigma * sqrt_vp * z1 + 0.25 * sigma2 * (z1 * z1 - dt)
            _record_log(i + 1, log_s)
        return np.exp(log_s) if _terminal_only else nodes

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
        _record_log(0, log_s)
        for i in range(n_steps):
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
                mgf = np.where(quad_mask, m_quad, m_exp)
                ln_M = np.log(mgf)
                K0 = -ros * kappa * theta * dt
                K0_star = -ln_M - (K1 + 0.5 * K3) * v_n
                # replace K0 with K0* on top of the standard increment
                log_s = (log_s + (drift - 0.5 * v_bar) * dt + corr - K0 + K0_star
                         + np.sqrt(v_bar) * sqrt_dt * diff_coef * z_ind[:, i])
            else:
                # QUADEXP: preserve the exact original expression grouping (bit-identical seed pins).
                log_s = log_s + (drift - 0.5 * v_bar) * dt + corr + np.sqrt(v_bar) * sqrt_dt * diff_coef * z_ind[:, i]
            v_n = v_np
            _record_log(i + 1, log_s)
        return np.exp(log_s) if _terminal_only else nodes

    raise ValidationError(f"unknown Heston MC scheme: {scheme}")


def simulate_heston_spot_nodes(
    s0,
    params,
    scheme,
    step_dt,
    r_fwd,
    carry_fwd,
    z_var,
    z_ind,
    u_var=None,
    record_steps=None,
):
    """Evolve Heston paths and return only the requested spot nodes.

    ``record_steps`` uses node indices in ``[0, M]`` and defaults to every
    node. Supplying contractual node indices lets path-dependent engines use
    fine simulation substeps without retaining the full fine-grid path.
    """
    return _simulate_heston_spots(
        s0,
        params,
        scheme,
        step_dt,
        r_fwd,
        carry_fwd,
        z_var,
        z_ind,
        u_var,
        record_steps=record_steps,
    )


def _simulate_terminal_spot(
    s0, params, scheme, step_dt, r_fwd, carry_fwd, z_var, z_ind, u_var,
):
    """Evolve spot and variance to maturity for all paths; return terminal spot."""
    return _simulate_heston_spots(
        s0,
        params,
        scheme,
        step_dt,
        r_fwd,
        carry_fwd,
        z_var,
        z_ind,
        u_var,
        _terminal_only=True,
    )


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


def price_barrier_heston_mc(
    s0, strike, is_call, params, step_dt, r_fwd, carry_fwd, disc_factor,
    barrier, is_up, is_out, rebate=0.0, pay_at_hit=False, continuous=True,
    observe_idx=None, participation=1.0, num_paths=50_000, seed=42, return_stderr=False,
):
    """Single-barrier option under Heston via MC (log-Euler full-truncation path recorder).

    Uses the positivity-preserving EULERLOG scheme and records nodes + per-step effective
    vol sqrt(v_+) so the shared barrier core applies continuous (Brownian-bridge) or discrete
    monitoring. The European limit (barrier far) matches ``price_european_heston_mc`` under the
    same EULERLOG scheme. See ``quantark.volmodels.barrier`` for payoff/monitoring semantics.
    """
    from quantark.volmodels.barrier import (
        BarrierSpec, bridge_survival, discrete_survival, disc_closure, mc_barrier_cashflows, validate_barrier,
    )
    dt = np.asarray(step_dt, dtype=float)
    rf = np.asarray(r_fwd, dtype=float)
    cf = np.asarray(carry_fwd, dtype=float)
    n = dt.size
    if n < 1 or rf.size != n or cf.size != n:
        raise ValidationError("step_dt, r_fwd, carry_fwd must be equal-length, length >= 1")
    if s0 <= 0 or strike <= 0:
        raise ValidationError("s0 and strike must be positive")
    if num_paths <= 0:
        raise ValidationError("num_paths must be positive")
    spec = BarrierSpec(bool(is_up), bool(is_out), bool(is_call), float(barrier), float(strike),
                       float(rebate), bool(pay_at_hit))
    validate_barrier(spec, s0)
    if not continuous and observe_idx is None:
        raise ValidationError("discrete monitoring requires observe_idx")
    if continuous and pay_at_hit:
        raise ValidationError("pay_at_hit=True is not supported with continuous bridge MC; use discrete monitoring or the PDE engine")

    kappa, theta, sigma, rho, v0 = params.kappa, params.theta, params.sigma, params.rho, params.v0
    rho_hat = np.sqrt(max(1.0 - rho * rho, 0.0))
    sigma2 = sigma * sigma
    rng = np.random.default_rng(seed)

    nodes = np.empty((num_paths, n + 1), dtype=float)
    vols = np.empty((num_paths, n), dtype=float)
    log_s = np.full(num_paths, np.log(max(float(s0), 1e-12)))
    v = np.full(num_paths, max(float(v0), 0.0))
    nodes[:, 0] = np.exp(log_s)
    for i in range(n):
        h = dt[i]; sh = np.sqrt(h)
        z1 = rng.standard_normal(num_paths) * sh
        z2 = rng.standard_normal(num_paths) * sh
        z_s = rho * z1 + rho_hat * z2
        vp = np.maximum(v, 0.0); svp = np.sqrt(vp)
        vols[:, i] = svp
        log_s = log_s + (rf[i] - cf[i] - 0.5 * vp) * h + svp * z_s
        v = v + kappa * (theta - vp) * h + sigma * svp * z1 + 0.25 * sigma2 * (z1 * z1 - h)
        nodes[:, i + 1] = np.exp(log_s)

    disc, node_times = disc_closure(dt, rf)
    T = float(node_times[-1])
    if continuous:
        w, first = bridge_survival(nodes, vols, dt, spec)
        hit_cumT = node_times[np.minimum(first, n)]
    else:
        idx = np.asarray(observe_idx, dtype=int)
        w, first = discrete_survival(nodes[:, idx], spec)
        hit_cumT = node_times[idx[np.minimum(first, idx.size - 1)]]

    pv = mc_barrier_cashflows(nodes[:, -1], w, hit_cumT, spec, disc, T, participation=participation)
    price = float(np.mean(pv))
    if return_stderr:
        return price, (float(np.std(pv, ddof=1) / np.sqrt(num_paths)) if num_paths > 1 else 0.0)
    return price
