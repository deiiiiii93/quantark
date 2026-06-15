"""Forward Fokker-Planck leverage calibration entry point.

Marches the log-variance joint density forward, reading the leverage off each slice, and
assembles a LeverageSurface. The eta*sigma == 0 case (deterministic variance) is the exact
local-volatility limit and is handled analytically (no PDE).
"""

from __future__ import annotations

import numpy as np

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface
from quantark.volmodels.slv.slv_mc_kernel import _validate_common
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig
from quantark.volmodels.slv.fokkerplanck.fp_solver import ForwardFPADI
from quantark.volmodels.slv.fokkerplanck.bootstrap import leverage_from_slice


def calibrate_leverage_surface_fp(s0, params: HestonParams, lv_surface, step_dt, r_fwd, carry_fwd,
                                  eta: float = 1.0, config: FpCalibrationConfig = None) -> LeverageSurface:
    """Calibrate the SLV leverage surface by forward Fokker-Planck (deterministic; no MC)."""
    config = config or FpCalibrationConfig()
    dt, rf, cf = _validate_common(s0, s0, step_dt, r_fwd, carry_fwd, num_paths=2, num_bins=2, eta=eta)
    if params.v0 <= 0.0:
        raise ValidationError("FFP requires v0 > 0")
    # rannacher_steps need no upper bound: the loop's `n < rannacher_steps` self-clamps (extra
    # implicit start-up steps are harmless), and the deterministic branch ignores them entirely.
    t_nodes = np.concatenate([[0.0], np.cumsum(dt)])
    record_times = t_nodes[:-1]                            # t_0 .. t_{M-1}

    sig_eff = eta * params.sigma
    if sig_eff == 0.0:                                     # exact deterministic-variance branch (no PDE)
        return _deterministic_surface(s0, params, lv_surface, record_times, config)
    # stochastic-path requirements (log-variance grid + positive stationary exponent)
    if params.kappa <= 0:
        raise ValidationError("stochastic FFP requires kappa > 0")
    if params.theta <= config.v_floor or params.v0 <= config.v_floor:
        raise ValidationError("stochastic FFP requires v0, theta > v_floor")

    sup_lv2 = float(np.max(lv_surface.lv_grid) ** 2)
    vbar2 = max(params.v0, params.theta, sup_lv2)
    b_steps = rf - cf
    solver = ForwardFPADI.from_config(s0, params, eta, b=float(np.mean(b_steps)),
                                      step_dt=dt, config=config, vbar2=vbar2, b_steps=b_steps)

    S_nodes = np.exp(solver.x)
    rows, mass_res, n_blend, n_clip = [], [], 0, 0
    f = solver.seed_dirac(s0, params.v0)
    for n in range(dt.size):
        t = record_times[n]
        sigma_lv_x = np.asarray(lv_surface.local_vol(S_nodes, t), float)
        if n == 0:
            L_raw = sigma_lv_x / np.sqrt(params.v0)        # exact t->0+ seed row
            lo, hi = config.leverage_clip
            L = np.clip(L_raw, lo, hi)
            diag = {"n_tail_blended": 0, "n_clipped": int(np.sum(L != L_raw)), "mass_residual": 0.0}
        else:
            L, diag = leverage_from_slice(solver, f, sigma_lv_x, t, params, eta, config)
        rows.append(L)
        mass_res.append(diag["mass_residual"])
        n_blend += diag["n_tail_blended"]
        n_clip += diag["n_clipped"]
        f = solver.step(f, L, dt[n], implicit=(n < config.rannacher_steps), b=float(rf[n] - cf[n]))
        m = solver.total_mass(f)
        if abs(m - 1.0) > config.mass_tol:
            raise NumericalError(f"FP density lost mass {m:.6f} at t={t_nodes[n + 1]:.4f} "
                                 f"(refine grid / extents)")

    sel = np.unique(np.linspace(0, solver.x.size - 1, config.n_strike_nodes).round().astype(int))
    strike_grid = S_nodes[sel]
    leverage_grid = np.vstack([row[sel] for row in rows])
    diagnostics = {"mass_residual": mass_res, "n_tail_blended": n_blend, "n_clipped": n_clip,
                   "method": "forward_fokker_planck"}
    return LeverageSurface(time_grid=record_times, strike_grid=strike_grid,
                           leverage_grid=leverage_grid, diagnostics=diagnostics)


def _deterministic_surface(s0, params: HestonParams, lv_surface, record_times,
                           config: FpCalibrationConfig) -> LeverageSurface:
    """eta*sigma == 0: variance is deterministic nu_det(t)=theta+(v0-theta)e^{-k t}; L=sigma_LV/sqrt(nu_det)."""
    if params.v0 <= 0.0:
        raise ValidationError("deterministic branch requires v0 > 0")
    x_lo, x_hi = np.log(s0) - 4.0, np.log(s0) + 4.0       # local-vol surface is the only vol scale here
    S_nodes = np.exp(np.linspace(x_lo, x_hi, config.n_strike_nodes))
    lo, hi = config.leverage_clip
    rows, n_clip = [], 0
    for t in record_times:
        nu_det = params.theta + (params.v0 - params.theta) * np.exp(-params.kappa * t)
        if nu_det <= 0:
            raise ValidationError("deterministic variance hit zero; invalid params for eta=0 branch")
        L_raw = np.asarray(lv_surface.local_vol(S_nodes, t), float) / np.sqrt(nu_det)
        L = np.clip(L_raw, lo, hi)
        n_clip += int(np.sum(L != L_raw))
        rows.append(L)
    return LeverageSurface(time_grid=np.asarray(record_times, float), strike_grid=S_nodes,
                           leverage_grid=np.vstack(rows),
                           diagnostics={"method": "deterministic_eta0", "n_clipped": n_clip})
