"""Barrier-local continuous-KI crossing coefficients for the vol-model PDEs.

The FIRST_PASSAGE correction (see
``SnowballPDESolver._prepare_continuous_ki_correction``) reduces the intra-step
barrier crossing to a per-step-constant GBM. That reduction is BARRIER-LOCAL --
the closed form never looks further than a few step-widths either side of the
barrier -- so a solver qualifies as soon as it can report the diffusion and
drift THERE. It does not need globally constant coefficients, which is what the
vol-model solvers were once gated off the correction for.

The mixins here report those coefficients for the two vol-model families. They
are shared by the Snowball and Phoenix solvers, which run the same two-surface
knock-in dynamic programming on different payoffs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.asset.equity.engine.pde.snowball_pde_solver import (
    _ContinuousKIFirstPassage,
)
from quantark.util.enum.engine_enums import ContinuousKICorrection
from quantark.util.exceptions import PricingError, ValidationError


class LocalVolBarrierCrossingMixin:
    """Local vol AT THE BARRIER, sampled at each step's midpoint."""

    def _first_passage_step_coefficients(self, product, pricing_env, t_vec, barrier):
        """``sigma_loc(barrier, t)`` rather than the term vol at the strike.

        The crossing happens at the barrier, so that is where the diffusion
        governing it must be read. The midpoint convention matches
        ``_build_step_coefficients`` exactly, so the correction and the
        stepping operator sample the same surface at the same times. A flat
        surface makes local vol constant and this reduces to the base sampling.
        """
        surface = self._active_lv_surface
        if surface is None:
            raise PricingError("Local-vol surface is not initialized for this solve")

        from quantark.priceenv.term_sampling import TermCoefficients

        t = np.asarray(t_vec, dtype=float)
        tc = TermCoefficients.from_env(
            pricing_env, t, ref_strike=float(product.strike)
        )
        t_mid = 0.5 * (t[:-1] + t[1:])
        sigma = np.asarray(
            surface.local_vol(np.full(t_mid.shape, float(barrier)), t_mid),
            dtype=float,
        )
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValidationError(
                "local-vol surface returned non-positive or non-finite vols "
                "at the KI barrier"
            )
        sig2 = sigma * sigma
        return tc.fwd_rates - tc.fwd_carry - 0.5 * sig2, sig2


class Heston2DBarrierCrossingMixin:
    """One ``(mu, sigma^2)`` per variance column of the ADI surface."""

    def _barrier_leverage(self, barrier: float, t: float) -> float:
        """Leverage multiplying the spot diffusion at the barrier. 1 for Heston."""
        return 1.0

    def _prepare_2d_continuous_ki_correction(self, core, product) -> None:
        """Build the per-variance-column FIRST_PASSAGE state for the ADI sweep.

        Column ``k`` carries the spot dynamics CONDITIONAL on that column's
        variance. The ADI operator's own log-spot assembly uses diffusion
        ``0.5 * L(S,t)**2 * v`` and convection ``(r - q) - 0.5 * L(S,t)**2 * v``
        (``HestonSLVADICore._tri_S``), so the crossing coefficients at the
        barrier are ``sigma**2 = L(barrier,t)**2 * v`` and
        ``mu = (r - q) - 0.5 * sigma**2``, taken with the operator's own
        variance floor and forward-rate schedule. That keeps the correction
        consistent with what the scheme actually integrates -- the same
        discipline the 1-D solvers follow.

        Freezing the variance across the step neglects its own diffusion: an
        O(dt) error in ``sigma**2``, hence O(dt**1.5) in a correction that
        repairs O(sqrt(dt)). It is the same conditional-Gaussian reduction the
        Monte Carlo bridge makes.
        """
        self._ki_fp = None
        if not self._first_passage_ki_supported:
            return
        if not product.has_ki_barrier or not self._ki_continuous:
            return
        if self.params.continuous_ki_correction is not (
            ContinuousKICorrection.FIRST_PASSAGE
        ):
            return
        barrier = float(self._ki_barrier)
        if barrier <= 0.0:
            return

        n_t = int(core.N_T)
        dt = float(core.dt)
        if n_t < 2 or dt <= 0.0:
            return
        # The operator floors the variance grid identically; the v = 0 row
        # would otherwise hand the closed form a zero diffusion.
        v = np.maximum(np.asarray(core.V_grid, dtype=float), 1e-10)
        mu = np.empty((n_t, v.size), dtype=float)
        sig2 = np.empty((n_t, v.size), dtype=float)
        for j in range(n_t):
            t_mid = (j + 0.5) * dt
            r_step, q_step = core.forward_rates_at(t_mid)
            lev = float(self._barrier_leverage(barrier, t_mid))
            sig2[j, :] = (lev * lev) * v
            mu[j, :] = (r_step - q_step) - 0.5 * sig2[j, :]
        self._ki_fp = _ContinuousKIFirstPassage(
            dt=np.full(n_t, dt, dtype=float),
            mu=mu,
            sig2=sig2,
            is_reverse=bool(product.is_reverse),
        )

    def _first_passage_step_index(self, core, tau: float) -> Optional[int]:
        """Forward-time step index of the step that just landed on ``tau``.

        ``tau`` is time-to-maturity, so forward time is ``T - tau`` and the
        completed step spans ``[T - tau, T - tau + dt]``. Interior steps only,
        mirroring the 1-D guard: the maturity column has no step interior and
        the valuation column's events belong to the t=0 readout.
        """
        k = self._hook_tau_key(tau, float(core.dt))
        if k is None or not (0 < k < int(core.N_T)):
            return None
        return int(core.N_T) - k

    def _apply_continuous_ki_correction(self, U, core, tau, barrier, v1):
        """Restore the intra-step crossing mass the nodal mask cannot see.

        The mask monitors the continuous barrier only at step boundaries, so
        crossings INSIDE the step are invisible and the live surface is biased
        high by O(sqrt(dt)) (see ``SnowballPDESolver._apply_ki_jump``).
        """
        fp = self._ki_fp
        if fp is None:
            return U
        j = self._first_passage_step_index(core, tau)
        if j is None:
            return U
        return U + fp.step_correction(j, core.S_grid, float(barrier), v1 - U)


class SLVBarrierLeverageMixin:
    """Reads the calibrated leverage surface at the barrier."""

    def _barrier_leverage(self, barrier: float, t: float) -> float:
        """``eta`` scales the vol-of-vol, not the spot diffusion, so the
        crossing sees the raw calibrated leverage -- exactly what the ADI
        core's ``_L`` feeds its log-spot operator."""
        return float(
            np.asarray(self.leverage_surface.leverage(float(barrier), float(t)))
        )
