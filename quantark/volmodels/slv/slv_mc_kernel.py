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
                  leverage_surface=None, leverage_clip=DEFAULT_LEVERAGE_CLIP):
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
) -> Union[float, Tuple[float, float]]:
    """Price a European vanilla under Heston SLV via MC.

    Leverage is calibrated on-the-fly by default (clipped to ``leverage_clip``, the band
    shared with the FFP route). Supplying ``leverage_surface`` uses that precomputed
    artifact directly — consumed as-is, never re-clipped — which is required for
    reproducible structured model risk under frozen/recalibrated leverage conventions.
    """
    dt, rf, cf = _validate_common(s0, strike, step_dt, r_fwd, carry_fwd, num_paths, num_bins, eta)
    _validate_clip(leverage_clip)
    if not np.isfinite(disc_factor) or disc_factor <= 0:
        raise ValidationError("disc_factor must be finite and positive")
    if leverage_surface is not None and not isinstance(leverage_surface, LeverageSurface):
        raise ValidationError("leverage_surface must be a LeverageSurface when provided")
    rng = np.random.default_rng(seed)
    s_terminal, _, _ = _simulate_slv(s0, params, lv_surface, eta, dt, rf, cf,
                                     num_paths, num_bins, bin_method, rng,
                                     leverage_surface=leverage_surface,
                                     leverage_clip=leverage_clip)
    if not np.all(np.isfinite(s_terminal)):
        from quantark.util.exceptions import NumericalError
        raise NumericalError("SLV MC produced non-finite terminal spots")
    payoff = np.maximum(s_terminal - strike, 0.0) if is_call else np.maximum(strike - s_terminal, 0.0)
    discounted = float(disc_factor) * payoff
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
