"""Heston Stochastic-Local-Volatility Monte Carlo kernel.

Simulates spot + variance with a leverage L(S,t) = sigma_LV(S,t)/sqrt(E[v|S]) where
the conditional expectation is estimated on-the-fly by binning (van der Stoep,
Grzelak & Oosterlee 2014). The variance follows full-truncation Euler (CIR) with
vol-of-vol eta*sigma; spot and variance share the same correlated Brownian, so the
spot scheme is martingale-consistent up to the O(dt) Euler bias (QE was deliberately
avoided here — see _simulate_slv's docstring for the rationale). Asset-neutral
per-step forwards (carry = dividend yield / foreign rate).

Also provides calibrate_leverage_surface, which materializes the calibrated leverage
on a fixed (t, S) grid as a LeverageSurface for the deterministic backward SLV PDE.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.slv.leverage import (
    DEFAULT_LEVERAGE_CLIP,
    BinMethod,
    LeverageSurface,
    bin_conditional,
    eval_binned,
)

_KMIN = 1e-8


def _simulate_slv(s0, params, lv_surface, eta, step_dt, r_fwd, carry_fwd,
                  num_paths, num_bins, bin_method, rng, record_grid=None,
                  leverage_surface=None, leverage_clip=DEFAULT_LEVERAGE_CLIP,
                  use_antithetic=False, qmc_z=None):
    """Full-truncation log-Euler SLV with a shared correlated Brownian.

    Variance and the rho-correlated part of spot are driven by the SAME Brownian dW_v,
    so the spot scheme is martingale-consistent up to the O(dt) symmetric Euler bias
    (E[S_T] -> forward as the step count grows) — avoiding the QE correlation-
    reconstruction drift bias. The Dupire local vol is frozen at the start of each step
    (Euler), an O(dt) discretization choice. The leverage sigma_hat = sigma_LV /
    sqrt(E[v|S]) is calibrated on-the-fly by binning.
    """
    kappa, theta, sigma = params.kappa, params.theta, params.sigma
    rho = float(np.clip(params.rho, -0.999, 0.999))
    rho_bar = np.sqrt(max(1.0 - rho * rho, 0.0))
    sigma_eff = float(eta) * sigma
    lo, hi = leverage_clip
    M = step_dt.size

    log_s = np.full(num_paths, np.log(max(s0, 1e-12)))
    v = np.full(num_paths, max(params.v0, 0.0))
    t = 0.0
    records = []
    n_clip_records = 0

    for i in range(M):
        dt = step_dt[i]
        sqrt_dt = np.sqrt(dt)
        drift_i = r_fwd[i] - carry_fwd[i]
        S = np.exp(log_s)

        if leverage_surface is None:
            boundaries, bin_means = bin_conditional(S, v, num_bins, bin_method)
            econd = np.maximum(eval_binned(S, boundaries, bin_means), _KMIN)
            sigma_lv = np.asarray(lv_surface.local_vol(S, t), dtype=float)
            sigma_hat = np.clip(sigma_lv / np.sqrt(econd), lo, hi)
            sigma_hat2 = sigma_hat * sigma_hat

            if record_grid is not None:
                econd_nodes = np.maximum(eval_binned(record_grid, boundaries, bin_means), _KMIN)
                lv_nodes = np.asarray(lv_surface.local_vol(record_grid, t), dtype=float)
                # Same clip as the in-simulation sigma_hat so the recorded leverage matches
                # the effective leverage the MC used (consumed identically by the backward PDE).
                raw_nodes = lv_nodes / np.sqrt(econd_nodes)
                clipped_nodes = np.clip(raw_nodes, lo, hi)
                n_clip_records += int(np.sum(clipped_nodes != raw_nodes))
                records.append(clipped_nodes)
        else:
            sigma_hat = np.asarray(leverage_surface.leverage(S, t), dtype=float)
            sigma_hat2 = sigma_hat * sigma_hat
            if not np.all(np.isfinite(sigma_hat2)):
                raise ValidationError("precomputed leverage returned non-finite values")
            if record_grid is not None:
                records.append(np.asarray(leverage_surface.leverage(record_grid, t), dtype=float))

        v_plus = np.maximum(v, 0.0)
        sqrt_vp = np.sqrt(v_plus)
        if qmc_z is not None:
            # QMC: low-discrepancy normals, column 0 = variance, column 1 = spot-independent.
            dW_v = sqrt_dt * qmc_z[:, 0, i]
            dW_s = rho * dW_v + rho_bar * sqrt_dt * qmc_z[:, 1, i]
        elif use_antithetic:
            # First half drawn, second half is its antithetic mirror (both z-streams).
            # z draws happen exactly here, so use_antithetic=False leaves the stream unchanged.
            half = num_paths // 2
            z_v_h = rng.standard_normal(half)
            z_i_h = rng.standard_normal(half)
            dW_v = sqrt_dt * np.concatenate([z_v_h, -z_v_h])
            dW_s = rho * dW_v + rho_bar * sqrt_dt * np.concatenate([z_i_h, -z_i_h])
        else:
            dW_v = sqrt_dt * rng.standard_normal(num_paths)
            dW_s = rho * dW_v + rho_bar * sqrt_dt * rng.standard_normal(num_paths)

        # spot: martingale log-Euler with leverage-adjusted vol sigma_hat*sqrt(v)
        log_s = np.maximum(
            log_s + (drift_i - 0.5 * sigma_hat2 * v_plus) * dt + sigma_hat * sqrt_vp * dW_s,
            np.log(1e-12),
        )
        # variance: full-truncation Euler (CIR) with vol-of-vol eta*sigma
        v = v + kappa * (theta - v_plus) * dt + sigma_eff * sqrt_vp * dW_v
        t += dt

    return np.exp(log_s), records, n_clip_records


def _validate_clip(leverage_clip) -> None:
    lo, hi = leverage_clip
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
        raise ValidationError("leverage_clip must be a finite positive ordered (lo, hi) tuple")


def _validate_common(s0, strike, step_dt, r_fwd, carry_fwd, num_paths, num_bins, eta):
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
    if num_paths <= 0 or num_bins < 2:
        raise ValidationError("num_paths must be positive and num_bins >= 2")
    if eta < 0:
        raise ValidationError("eta must be non-negative")
    return dt, rf, cf


def price_european_slv_mc(
    s0: float, strike: float, is_call: bool, params: HestonParams, lv_surface,
    step_dt: np.ndarray, r_fwd: np.ndarray, carry_fwd: np.ndarray, disc_factor: float,
    eta: float = 1.0, num_paths: int = 50_000, num_bins: int = 20,
    bin_method: BinMethod = BinMethod.EQUAL_WEIGHTED, seed: Optional[int] = 42,
    return_stderr: bool = False, leverage_surface: Optional[LeverageSurface] = None,
    leverage_clip: Tuple[float, float] = DEFAULT_LEVERAGE_CLIP,
    use_antithetic: bool = False,
    sampler=None,
) -> Union[float, Tuple[float, float]]:
    """Price a European vanilla under Heston SLV via MC.

    Leverage is calibrated on-the-fly by default (clipped to ``leverage_clip``, the band
    shared with the FFP route). Supplying ``leverage_surface`` uses that precomputed
    artifact directly — consumed as-is, never re-clipped — which is required for
    reproducible structured model risk under frozen/recalibrated leverage conventions.

    sampler (optional): a quantark.montecarlo generator exposing ``uniform(n, dim)``.
        QMC dimension layout: columns [z_var(M) | z_ind(M)] reshaped to (n, 2, M) after
        ndtri (col 0 = variance normal, col 1 = spot-independent). Mutually exclusive with
        ``use_antithetic``; default None keeps the pseudo path bit-identical.
    """
    dt, rf, cf = _validate_common(s0, strike, step_dt, r_fwd, carry_fwd, num_paths, num_bins, eta)
    _validate_clip(leverage_clip)
    if not np.isfinite(disc_factor) or disc_factor <= 0:
        raise ValidationError("disc_factor must be finite and positive")
    if leverage_surface is not None and not isinstance(leverage_surface, LeverageSurface):
        raise ValidationError("leverage_surface must be a LeverageSurface when provided")
    M = dt.size
    qmc_z = None
    if sampler is not None:
        if use_antithetic:
            raise ValidationError("sampler and use_antithetic are mutually exclusive")
        from scipy.special import ndtri
        raw = np.clip(np.asarray(sampler.uniform(num_paths, 2 * M), dtype=float),
                      1e-12, 1.0 - 1e-12)
        qmc_z = ndtri(raw).reshape(num_paths, 2, M)   # [:,0,:]=z_var, [:,1,:]=z_ind
    rng = np.random.default_rng(seed)
    # Antithetic: simulate 2*half paths (half originals + half mirrors), then average each
    # pair's discounted payoff — mirroring the Heston MC convention. Default off is an
    # unchanged z-stream (bit-identical to the pre-change kernel).
    half = (num_paths + 1) // 2
    n_eff = num_paths if sampler is not None else (2 * half if use_antithetic else num_paths)
    s_terminal, _, _ = _simulate_slv(s0, params, lv_surface, eta, dt, rf, cf,
                                     n_eff, num_bins, bin_method, rng,
                                     leverage_surface=leverage_surface,
                                     leverage_clip=leverage_clip,
                                     use_antithetic=use_antithetic, qmc_z=qmc_z)
    if not np.all(np.isfinite(s_terminal)):
        from quantark.util.exceptions import NumericalError
        raise NumericalError("SLV MC produced non-finite terminal spots")
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


def _calibrate_mc_binning(
    s0: float, params: HestonParams, lv_surface,
    step_dt: np.ndarray, r_fwd: np.ndarray, carry_fwd: np.ndarray,
    eta: float = 1.0, num_paths: int = 50_000, num_bins: int = 20,
    bin_method: BinMethod = BinMethod.EQUAL_WEIGHTED, seed: Optional[int] = 42,
    n_strike_nodes: int = 41, strike_span_stds: float = 4.0,
    leverage_clip: Tuple[float, float] = DEFAULT_LEVERAGE_CLIP,
) -> LeverageSurface:
    """Materialize the SLV leverage L(S,t) on a fixed (t, S) grid via MC binning.

    The leverage is recorded at each simulation time node on a fixed log-spaced strike
    grid spanning +/- strike_span_stds total-vol standard deviations around s0, clipped
    to ``leverage_clip`` (the band shared with the FFP route).
    """
    dt, rf, cf = _validate_common(s0, strike=s0, step_dt=step_dt, r_fwd=r_fwd,
                                  carry_fwd=carry_fwd, num_paths=num_paths, num_bins=num_bins, eta=eta)
    _validate_clip(leverage_clip)
    T = float(dt.sum())
    width = strike_span_stds * np.sqrt(max(params.theta, params.v0, 0.04) * max(T, 1e-12))
    strike_grid = s0 * np.exp(np.linspace(-width, width, n_strike_nodes))
    rng = np.random.default_rng(seed)
    _, records, n_clip = _simulate_slv(s0, params, lv_surface, eta, dt, rf, cf,
                                       num_paths, num_bins, bin_method, rng,
                                       record_grid=strike_grid, leverage_clip=leverage_clip)
    # records[i] is recorded at the START of step i, i.e. at time node t_i (t_0 = 0).
    # There are M records at t_0 .. t_{M-1}; the leverage at t in (t_{M-1}, T] is covered
    # by flat extrapolation in the LeverageSurface (the backward PDE starts at T from the payoff).
    record_times = np.concatenate([[0.0], np.cumsum(dt)])[:-1]  # t_0 .. t_{M-1}
    leverage_grid = np.vstack(records)
    return LeverageSurface(time_grid=record_times, strike_grid=strike_grid,
                           leverage_grid=leverage_grid,
                           diagnostics={"method": "mc_binning", "n_clipped": int(n_clip)})


_UNSET = object()   # sentinel: distinguishes "MC option not provided" from an explicit value (e.g. seed=None)


def calibrate_leverage_surface(
    s0, params, lv_surface, step_dt, r_fwd, carry_fwd, eta=1.0,
    num_paths=_UNSET, num_bins=_UNSET, bin_method=_UNSET, seed=_UNSET,
    n_strike_nodes=_UNSET, strike_span_stds=_UNSET, leverage_clip=_UNSET,
    *, method=None, fp_config=None,
):
    """Dispatch leverage calibration. Default: forward Fokker-Planck (deterministic).

    The MC options keep their original positional slots after ``eta`` (so old positional calls bind
    correctly), while ``method``/``fp_config`` are keyword-only. MC-binning is retained as an
    independent cross-check (method=MC_BINNING). Method-specific options are validated, never silently
    dropped: any MC option under FFP raises, and a stray fp_config under MC_BINNING raises. NOTE: with
    the FFP default, MC options require an explicit ``method=MC_BINNING`` (the deliberate consequence of
    the default flip + mismatch rejection); they no longer auto-select MC.
    """
    from quantark.util.enum.engine_enums import LeverageCalibrationMethod
    if method is None:
        method = LeverageCalibrationMethod.FORWARD_FOKKER_PLANCK
    mc_opts = {k: v for k, v in dict(
        num_paths=num_paths, num_bins=num_bins, bin_method=bin_method, seed=seed,
        n_strike_nodes=n_strike_nodes, strike_span_stds=strike_span_stds,
        leverage_clip=leverage_clip,
    ).items() if v is not _UNSET}
    if method is LeverageCalibrationMethod.MC_BINNING:
        if fp_config is not None:
            raise ValidationError("fp_config is not valid for MC_BINNING")
        return _calibrate_mc_binning(s0, params, lv_surface, step_dt, r_fwd, carry_fwd,
                                     eta=eta, **mc_opts)
    if method is LeverageCalibrationMethod.FORWARD_FOKKER_PLANCK:
        if mc_opts:
            raise ValidationError(
                f"MC options {sorted(mc_opts)} are not valid for FORWARD_FOKKER_PLANCK; "
                f"pass method=LeverageCalibrationMethod.MC_BINNING to use MC binning")
        from quantark.volmodels.slv.fokkerplanck.calibration import calibrate_leverage_surface_fp
        return calibrate_leverage_surface_fp(s0, params, lv_surface, step_dt, r_fwd, carry_fwd,
                                             eta=eta, config=fp_config)
    raise ValidationError("UNCONDITIONAL_MEAN is not a calibration method")


def price_barrier_slv_mc(
    s0, strike, is_call, params, leverage_surface, step_dt, r_fwd, carry_fwd, disc_factor,
    barrier, is_up, is_out, rebate=0.0, pay_at_hit=False, continuous=True, eta=1.0,
    observe_idx=None, participation=1.0, num_paths=50_000, seed=42, return_stderr=False,
):
    """Single-barrier option under Heston-SLV via MC using a precomputed leverage surface.

    Full-truncation log-Euler mirroring ``_simulate_slv`` (precomputed-leverage branch); the
    per-step effective vol is L(S,t)*sqrt(v_+), recorded together with the path nodes so the
    shared barrier core applies continuous (Brownian-bridge) or discrete monitoring.
    """
    from quantark.volmodels.barrier import (
        BarrierSpec, bridge_survival, discrete_survival, disc_closure, mc_barrier_cashflows, validate_barrier,
    )
    from quantark.volmodels.slv.leverage import LeverageSurface
    dt = np.asarray(step_dt, dtype=float); rf = np.asarray(r_fwd, dtype=float); cf = np.asarray(carry_fwd, dtype=float)
    n = dt.size
    if n < 1 or rf.size != n or cf.size != n:
        raise ValidationError("step_dt, r_fwd, carry_fwd must be equal-length, length >= 1")
    if not isinstance(leverage_surface, LeverageSurface):
        raise ValidationError("price_barrier_slv_mc requires a precomputed LeverageSurface")
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

    kappa, theta, sigma = params.kappa, params.theta, params.sigma
    rho = float(np.clip(params.rho, -0.999, 0.999))
    rho_bar = np.sqrt(max(1.0 - rho * rho, 0.0))
    sigma_eff = float(eta) * sigma
    rng = np.random.default_rng(seed)

    nodes = np.empty((num_paths, n + 1), dtype=float)
    vols = np.empty((num_paths, n), dtype=float)
    log_s = np.full(num_paths, np.log(max(float(s0), 1e-12)))
    v = np.full(num_paths, max(params.v0, 0.0))
    nodes[:, 0] = np.exp(log_s)
    t = 0.0
    for i in range(n):
        h = dt[i]; sh = np.sqrt(h)
        S = np.exp(log_s)
        sigma_hat = np.asarray(leverage_surface.leverage(S, t), dtype=float)
        vp = np.maximum(v, 0.0); svp = np.sqrt(vp)
        eff = sigma_hat * svp                        # effective instantaneous vol on this step
        vols[:, i] = eff
        dW_v = sh * rng.standard_normal(num_paths)
        dW_s = rho * dW_v + rho_bar * sh * rng.standard_normal(num_paths)
        log_s = np.maximum(log_s + (rf[i] - cf[i] - 0.5 * eff * eff) * h + eff * dW_s, np.log(1e-12))
        v = v + kappa * (theta - vp) * h + sigma_eff * svp * dW_v
        nodes[:, i + 1] = np.exp(log_s)
        t += h

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
