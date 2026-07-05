"""Forward Fokker-Planck solver for the SLV log-variance density.

v1 march is fully-coupled backward-Euler (unconditionally stable, damps the singular Dirac
seed). Task 5b adds Craig-Sneyd as the faster default with a Rannacher (implicit) start-up.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, bicgstab, LinearOperator

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig
from quantark.volmodels.slv.fokkerplanck.coordinates import (
    concentrated_grid, trapezoid_weights, x_extents, z_extents,
)
from quantark.volmodels.slv.fokkerplanck.fp_operators import build_directional_operators


class _MarchState:
    """Carries the lagged splu preconditioner for the krylov_lagged linear solver across march steps."""
    __slots__ = ("precond", "steps_since_refactor")

    def __init__(self):
        self.precond = None
        self.steps_since_refactor = 0


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
        """Discrete unit-mass seed at (ln s0, ln v0).

        ``seed_split=False`` (default): all mass on the nearest node, f(node) = 1/w_node (legacy).
        ``seed_split=True``: bilinear split over the <=4 bracketing nodes with weights divided by their
        quadrature weights -- total mass exactly 1, seed mean location exact to O(h^2) (removes the
        legacy O(h) placement bias). Independent of any flux scheme.
        """
        f = np.zeros(self.nx * self.nz)
        xs, zs = float(np.log(s0)), float(np.log(v0))
        if not self.cfg.seed_split:
            i = int(np.argmin(np.abs(self.x - xs)))
            j = int(np.argmin(np.abs(self.z - zs)))
            k = i * self.nz + j
            f[k] = 1.0 / self.w[k]
            return f
        i = int(np.clip(np.searchsorted(self.x, xs) - 1, 0, self.nx - 2))
        j = int(np.clip(np.searchsorted(self.z, zs) - 1, 0, self.nz - 2))
        tx = np.clip((xs - self.x[i]) / (self.x[i + 1] - self.x[i]), 0.0, 1.0)
        tz = np.clip((zs - self.z[j]) / (self.z[j + 1] - self.z[j]), 0.0, 1.0)
        for di, wx in ((0, 1.0 - tx), (1, tx)):
            for dj, wz in ((0, 1.0 - tz), (1, tz)):
                k = (i + di) * self.nz + (j + dj)
                f[k] += (wx * wz) / self.w[k]
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

    def _solve_implicit(self, M, rhs, state, *, advance=True):
        """Solve M x = rhs. ``direct``: splu. ``krylov_lagged``: BiCGStab + lagged splu preconditioner.

        ``advance``: when True (default) this solve participates in the refactor cadence (may refresh the
        preconditioner at the ``refactor_every`` boundary and increments the step counter). Pass
        ``advance=False`` for TR-BDF2's second substep, which reuses the first substep's identical M.
        """
        if self.cfg.linear_solver == "direct":
            return self._splu(M).solve(rhs)
        # krylov_lagged: BiCGStab against a lagged splu preconditioner refreshed every refactor_every steps
        if state.precond is None or (advance and state.steps_since_refactor >= self.cfg.refactor_every):
            state.precond = self._splu(M)
            state.steps_since_refactor = 0
        Mpre = LinearOperator(M.shape, matvec=state.precond.solve)
        x, info = bicgstab(M, rhs, M=Mpre, rtol=1e-13, atol=0.0, maxiter=300)
        if info != 0:                                    # one refactor + retry, then raise (no silent degrade)
            state.precond = self._splu(M)
            state.steps_since_refactor = 0
            Mpre = LinearOperator(M.shape, matvec=state.precond.solve)
            x, info = bicgstab(M, rhs, M=Mpre, rtol=1e-13, atol=0.0, maxiter=300)
            if info != 0:
                raise NumericalError(f"Krylov FP solve failed to converge (info={info}); refine grid/steps")
        if advance:
            state.steps_since_refactor += 1
        return x

    def step(self, f, L, dt, implicit=True, theta=0.5, b=None, state=None):
        """Advance the density one step.

        ``implicit=True`` (default) does a fully-coupled backward-Euler solve: L-stable and
        positivity-near-preserving. This is the scheme the calibration uses, because the ADI variant
        below is von-Neumann-unstable for this Dirac-seeded correlated *forward* density (negative
        mass grows under time refinement, to 1e7+ for strong correlation).

        ``implicit=False`` is a faster damped double-sweep ADI step (explicit predictor + directional
        implicit correctors + a mixed correction off ``Y2``, then a second pair of sweeps). It is
        retained for experimentation only and is NOT used for production calibration. ``b`` overrides
        the per-step cost-of-carry.
        """
        L = np.asarray(L, float)
        b_eff = self.b if b is None else float(b)
        Ax, Az, Axz = build_directional_operators(self.x, self.z, L, self.params, self.eta, b_eff)
        if implicit:
            if state is None:
                state = _MarchState()
            M = (self._I - dt * (Ax + Az + Axz)).tocsc()
            out = self._solve_implicit(M, f, state)
        else:
            A = Ax + Az + Axz
            lu_x = self._splu(self._I - theta * dt * Ax)
            lu_z = self._splu(self._I - theta * dt * Az)
            axf = theta * dt * (Ax @ f)
            azf = theta * dt * (Az @ f)
            Y0 = f + dt * (A @ f)                            # explicit predictor (full operator)
            Y1 = lu_x.solve(Y0 - axf)                        # implicit in x (subtract explicit Ax part)
            Y2 = lu_z.solve(Y1 - azf)                        # implicit in z
            Ycorr = Y2 + 0.5 * dt * (Axz @ (Y2 - f))         # damped mixed correction off Y2 (positivity)
            Z1 = lu_x.solve(Ycorr - axf)
            out = lu_z.solve(Z1 - azf)
        if not np.all(np.isfinite(out)):
            raise NumericalError("forward FP step produced non-finite density")
        return out

    _TRBDF2_GAMMA = 2.0 - np.sqrt(2.0)

    def march_step(self, f, L, dt, b, state, is_first):
        """Advance one step under cfg.time_scheme.

        ``backward_euler``: identical to ``step(implicit=True, state=state)`` (L-stable, damps the seed).
        ``tr_bdf2``: second-order L-stable one-step scheme (gamma=2-sqrt2). ``is_first=True`` forces a
        backward-Euler step (Rannacher-style start-up) to damp the singular Dirac seed. Because
        gamma=2-sqrt2 makes ``1/2*gamma == (1-gamma)/(2-gamma)``, the TR and BDF2 substeps share the SAME
        implicit operator M -- one factorization / one lagged preconditioner serves both solves.
        """
        if self.cfg.time_scheme == "backward_euler" or is_first:
            return self.step(f, L, dt, implicit=True, b=b, state=state)
        g = self._TRBDF2_GAMMA
        L = np.asarray(L, float)
        Ax, Az, Axz = build_directional_operators(self.x, self.z, L, self.params, self.eta, float(b))
        A = (Ax + Az + Axz).tocsc()
        M = (self._I - 0.5 * g * dt * A).tocsc()          # SAME matrix serves both substeps
        # TR substep to t + gamma*dt
        y_g = self._solve_implicit(M, f + 0.5 * g * dt * (A @ f), state, advance=True)
        # BDF2 substep to t + dt (reuse M/preconditioner: advance=False)
        c1 = 1.0 / (g * (2.0 - g))
        c0 = (1.0 - g) ** 2 / (g * (2.0 - g))
        out = self._solve_implicit(M, c1 * y_g - c0 * f, state, advance=False)
        if not np.all(np.isfinite(out)):
            raise NumericalError("TR-BDF2 FP step produced non-finite density")
        return out
