"""Per-slice leverage read-off for the forward Fokker-Planck calibration.

L(x, t)^2 = sigma_LV(e^x, t)^2 / E[v | x, t], with E[v|x] = (sum_j e^z f w_z)/(sum_j f w_z).
Deep tails (negligible marginal mass) blend smoothly to the unconditional CIR mean to avoid
0/0 noise and finite-difference kinks; result is clipped to a positive band.
"""

from __future__ import annotations

import numpy as np

from quantark.util.exceptions import NumericalError
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig


def conditional_variance(solver, f, cfg: FpCalibrationConfig) -> np.ndarray:
    """E[v | x] = sum_j e^z f w_z / sum_j f w_z, guarded against 0/0."""
    F = f.reshape(solver.nx, solver.nz)
    m = F @ solver.wz                                  # marginal mass per x
    num = F @ (solver.wz * solver.nu)                  # sum e^z f w_z
    return num / np.maximum(m, cfg.eps_mass)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def leverage_from_slice(solver, f, sigma_lv_x, t, params: HestonParams, eta, cfg):
    """Leverage row L(x) at time t with smooth mass-weighted tail blend + clip. Returns (L, diag)."""
    F = f.reshape(solver.nx, solver.nz)
    m = F @ solver.wz
    e_cond = conditional_variance(solver, f, cfg)
    e_uncond = params.theta + (params.v0 - params.theta) * np.exp(-params.kappa * t)
    omega = _smoothstep(m / (cfg.tail_mass_floor * max(m.max(), cfg.eps_mass)))
    e_blend = omega * e_cond + (1.0 - omega) * e_uncond
    if np.any(e_blend <= 0):
        raise NumericalError("non-positive blended conditional variance in leverage read-off")
    L = np.asarray(sigma_lv_x, float) / np.sqrt(e_blend)
    lo, hi = cfg.leverage_clip
    L_clipped = np.clip(L, lo, hi)
    diag = {
        "n_tail_blended": int(np.sum(omega < 0.999)),
        "n_clipped": int(np.sum(L != L_clipped)),
        "mass_residual": float(abs(solver.total_mass(f) - 1.0)),
    }
    if not np.all(np.isfinite(L_clipped)):
        raise NumericalError("non-finite leverage in read-off")
    return L_clipped, diag
