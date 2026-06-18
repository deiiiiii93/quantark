from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Tuple

from quantark.util.enum.engine_enums import ADIScheme
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class FpCalibrationConfig:
    """Configuration for forward Fokker-Planck leverage calibration.

    step_dt / r_fwd / carry_fwd are NOT here -- they are per-trade entry-point arguments.
    """
    n_x: int = 201
    n_z: int = 101
    x_span_stds: float = 6.0          # N_x in W_x = N_x*sqrt(vbar^2 * T), vbar^2 = max(v0, theta, sup sigma_LV^2)
    cir_quantile: float = 1e-4        # z-extent from CIR transition quantiles [q, 1-q] over t>0 nodes
    v_floor: float = 1e-8             # variance floor so z = ln v is finite (applied to CIR quantile only)
    x_concentration: float = 0.1      # sinh width around ln s0
    z_concentration: float = 0.1      # sinh width around ln v0
    n_strike_nodes: int = 41          # output LeverageSurface strike resolution (subsampled from x-grid)
    scheme: ADIScheme = ADIScheme.CRAIG_SNEYD
    rannacher_steps: int = 2          # fully-implicit coupled start-up steps
    mass_tol: float = 1e-3            # per-slice |sum(w*f) - 1| budget (tripwire)
    tail_mass_floor: float = 1e-6     # relative cell-mass below which leverage blends to uncond. mean
    leverage_clip: Tuple[float, float] = (0.05, 20.0)
    tol_neg: float = 0.5              # max negative probability mass sum_k w_k*max(-f_k,0). The central
    #                                   mixed-derivative (correlation) term is inherently not positivity-
    #                                   preserving (~10-15% for typical |rho|); negatives are clamped in
    #                                   the read-off. This catches genuine divergence (mass ~ unit total).
    eps_mass: float = 1e-300          # absolute guard for the E[v|x] ratio denominator

    def __post_init__(self) -> None:
        if not (isinstance(self.n_x, Integral) and self.n_x >= 3):
            raise ValidationError("n_x must be an integer >= 3")
        if not (isinstance(self.n_z, Integral) and self.n_z >= 3):
            raise ValidationError("n_z must be an integer >= 3")
        if not (isinstance(self.n_strike_nodes, Integral) and 3 <= self.n_strike_nodes <= self.n_x):
            raise ValidationError("n_strike_nodes must be an integer in [3, n_x]")
        if not (0.0 < self.cir_quantile < 0.5):
            raise ValidationError("cir_quantile must be in (0, 0.5)")
        for name, val in (("x_span_stds", self.x_span_stds), ("v_floor", self.v_floor),
                          ("x_concentration", self.x_concentration),
                          ("z_concentration", self.z_concentration),
                          ("mass_tol", self.mass_tol), ("tail_mass_floor", self.tail_mass_floor),
                          ("tol_neg", self.tol_neg), ("eps_mass", self.eps_mass)):
            if not (math.isfinite(val) and val > 0.0):
                raise ValidationError(f"{name} must be finite and positive")
        if not (isinstance(self.rannacher_steps, Integral) and self.rannacher_steps >= 0):
            raise ValidationError("rannacher_steps must be a non-negative integer")
        if self.scheme is not ADIScheme.CRAIG_SNEYD:
            raise ValidationError("only ADIScheme.CRAIG_SNEYD is specified in v1")
        lo, hi = self.leverage_clip
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi):
            raise ValidationError("leverage_clip must be a positive ordered (lo, hi) tuple")
