"""Forward Fokker-Planck solver for the SLV log-variance density.

v1 march is fully-coupled backward-Euler (unconditionally stable, damps the singular Dirac
seed). Task 5b adds Craig-Sneyd as the faster default with a Rannacher (implicit) start-up.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig
from quantark.volmodels.slv.fokkerplanck.coordinates import (
    concentrated_grid, trapezoid_weights, x_extents, z_extents,
)
from quantark.volmodels.slv.fokkerplanck.fp_operators import build_directional_operators


class ForwardFPADI:
    def __init__(self, x, z, params: HestonParams, eta, b, config: FpCalibrationConfig):
        self.x, self.z = np.asarray(x, float), np.asarray(z, float)
        self.params, self.eta, self.b, self.cfg = params, float(eta), float(b), config
        self.nx, self.nz = self.x.size, self.z.size
        self.wx, self.wz = trapezoid_weights(self.x), trapezoid_weights(self.z)
        self.w = np.outer(self.wx, self.wz).ravel()
        self.nu = np.exp(self.z)
        self._I = sp.identity(self.nx * self.nz, format="csc")

    @classmethod
    def from_config(cls, s0, params: HestonParams, eta, b, step_dt,
                    config: FpCalibrationConfig, vbar2: float = None, b_steps=None):
        """Build grids/weights from config. ``b`` is the scalar carry stored as the step default;
        ``b_steps`` (per-step cost-of-carry) sizes the x-extent drift envelope when forwards vary."""
        sig_eff2 = (eta * params.sigma) ** 2
        if sig_eff2 <= 0:
            raise ValidationError("ForwardFPADI requires eta*sigma > 0")
        step_dt = np.asarray(step_dt, float)
        t_nodes = np.concatenate([[0.0], np.cumsum(step_dt)])
        v_min, v_max = z_extents(params, eta, t_nodes, config.cir_quantile, config.v_floor)
        z = concentrated_grid(np.log(v_min), np.log(v_max), np.log(params.v0), config.n_z,
                              config.z_concentration)
        if vbar2 is None:
            vbar2 = max(params.v0, params.theta)
        carry = np.full(step_dt.size, b) if b_steps is None else np.asarray(b_steps, float)
        b_fwd = carry - 0.5 * vbar2                      # per-step forward log-drift envelope
        x_min, x_max = x_extents(s0, b_fwd, step_dt, vbar2, config.x_span_stds)
        x = concentrated_grid(x_min, x_max, np.log(s0), config.n_x, config.x_concentration)
        return cls(x, z, params, eta, b, config)

    def seed_dirac(self, s0, v0):
        """Discrete unit-mass seed at the node nearest (ln s0, ln v0): f(node) = 1/w_node."""
        f = np.zeros(self.nx * self.nz)
        i = int(np.argmin(np.abs(self.x - np.log(s0))))
        j = int(np.argmin(np.abs(self.z - np.log(v0))))
        k = i * self.nz + j
        f[k] = 1.0 / self.w[k]
        return f

    def total_mass(self, f):
        return float(self.w @ f)

    def spot_marginal(self, f):
        """Marginal density in x = ln S: integral over z of f(x, z) dz."""
        return f.reshape(self.nx, self.nz) @ self.wz

    @staticmethod
    def _splu(M):
        try:
            return splu(M.tocsc())
        except RuntimeError as exc:                      # singular factor -> refine grid, never clamp
            raise NumericalError(f"forward FP factorization failed: {exc}")

    def step(self, f, L, dt, implicit=False, theta=0.5, b=None):
        """Advance the density one step.

        ``implicit=True`` does a fully-coupled backward-Euler solve (unconditionally stable; used
        for the Rannacher start-up to damp the singular Dirac). Otherwise a Craig-Sneyd ADI step:
        explicit predictor + two directional implicit correctors (each subtracting that direction's
        explicit contribution already in the predictor) + the mixed-term correction. Mirrors the
        backward SLV PDE _cs_step. ``b`` overrides the cost-of-carry for this step (per-step forwards).
        """
        L = np.asarray(L, float)
        b_eff = self.b if b is None else float(b)
        Ax, Az, Axz = build_directional_operators(self.x, self.z, L, self.params, self.eta, b_eff)
        if implicit:
            out = self._splu(self._I - dt * (Ax + Az + Axz)).solve(f)
        else:
            A = Ax + Az + Axz
            lu_x = self._splu(self._I - theta * dt * Ax)
            lu_z = self._splu(self._I - theta * dt * Az)
            axf = theta * dt * (Ax @ f)
            azf = theta * dt * (Az @ f)
            Y0 = f + dt * (A @ f)                            # explicit predictor (full operator)
            Y1 = lu_x.solve(Y0 - axf)                        # implicit in x (subtract explicit Ax part)
            Y2 = lu_z.solve(Y1 - azf)                        # implicit in z
            Ycorr = Y2 + 0.5 * dt * (Axz @ (Y2 - f))         # Craig-Sneyd mixed-term correction
            Z1 = lu_x.solve(Ycorr - axf)
            out = lu_z.solve(Z1 - azf)
        if not np.all(np.isfinite(out)):
            raise NumericalError("forward FP step produced non-finite density")
        return out
