"""SABR stochastic-volatility forward process.

Two time-discretization schemes (mirroring HestonMCScheme in the SLV/Heston MC
kernels):
- LOG_EULER: log-Euler on the shifted forward, exact GBM on alpha. All beta.
- QUADEXP:   Andersen-style conditional lognormal. alpha sampled exactly; the
             forward sampled from its conditional law given the vol path. EXACT
             for beta=1 (only the integrated-variance quadrature is approximate),
             so it tolerates coarse time grids. beta != 1 is deferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantark.util.enum.engine_enums import SABRMCScheme
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance, validate_positive


@dataclass
class SABRProcess:
    """SABR forward dynamics:  dF = alpha * (F+shift)^beta dW1,  dalpha = nu*alpha dW2,
    corr(dW1, dW2) = rho. The forward is driftless (martingale)."""

    f0: float
    alpha: float
    beta: float
    rho: float
    nu: float
    shift: float = 0.0

    def __post_init__(self) -> None:
        validate_positive(self.f0 + self.shift, name="f0+shift")
        validate_positive(self.alpha, name="alpha")
        if not 0.0 <= self.beta <= 1.0:
            raise ValidationError(f"beta must be in [0, 1], got {self.beta}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValidationError(f"rho must be in [-1, 1], got {self.rho}")
        if self.nu < 0.0:
            raise ValidationError(f"nu must be non-negative, got {self.nu}")

    def simulate(
        self, T: float, n_paths: int, n_steps: int,
        *, seed: Optional[int] = None, antithetic: bool = True,
        scheme: SABRMCScheme = SABRMCScheme.LOG_EULER,
    ) -> np.ndarray:
        """Simulate terminal forwards F_T. Returns shape (n_paths,)."""
        validate_positive(T, name="T")
        if not isinstance(scheme, SABRMCScheme):
            raise ValidationError("scheme must be a SABRMCScheme")
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        floor = Tolerance.LOG_MIN

        # Two correlated normal blocks; one extra independent block for QE's
        # orthogonal forward shock. Antithetic mirrors every draw.
        if antithetic:
            half = (n_paths + 1) // 2
            base = rng.standard_normal((3, half, n_steps))
            blk = np.concatenate([base, -base], axis=1)[:, :n_paths, :]
        else:
            blk = rng.standard_normal((3, n_paths, n_steps))

        if scheme == SABRMCScheme.LOG_EULER:
            # Log-space (geometric) step with the frozen local vol α·F̃^{β-1}:
            #   d ln F̃ = α F̃^{β-1} dW1 − ½ (α F̃^{β-1})² dt
            # positivity-preserving and exact GBM at β=1. alpha is exact GBM.
            z1 = blk[0]
            z2 = self.rho * blk[0] + np.sqrt(max(1.0 - self.rho**2, 0.0)) * blk[1]
            f = np.full(n_paths, self.f0 + self.shift, dtype=float)
            a = np.full(n_paths, self.alpha, dtype=float)
            for k in range(n_steps):
                local_vol = a * np.power(np.maximum(f, floor), self.beta - 1.0)
                f = np.maximum(
                    f * np.exp(local_vol * sqrt_dt * z1[:, k] - 0.5 * local_vol**2 * dt),
                    floor,
                )
                a = a * np.exp(self.nu * sqrt_dt * z2[:, k] - 0.5 * self.nu**2 * dt)
            return f - self.shift

        if scheme == SABRMCScheme.QUADEXP:
            if abs(self.beta - 1.0) > Tolerance.PRECISION:
                # TODO(sabr-qe-general-beta): implement the Islah (2009) /
                # Leitao-Grzelak-Oosterlee noncentral-chi-square conditional
                # moments + lognormal/QE moment-match. Do not fake it.
                raise NotImplementedError(
                    "QUADEXP SABR is implemented only for beta=1 (exact conditional "
                    "lognormal); use SABRMCScheme.LOG_EULER for beta != 1."
                )
            rho_hat = np.sqrt(max(1.0 - self.rho**2, 0.0))
            nu_zero = self.nu <= Tolerance.PRECISION
            z2 = blk[1]            # drives the (exact) alpha GBM
            z_perp = blk[2]        # orthogonal forward shock
            log_f = np.full(n_paths, np.log(self.f0 + self.shift), dtype=float)
            a = np.full(n_paths, self.alpha, dtype=float)
            for k in range(n_steps):
                a_next = a * np.exp(self.nu * sqrt_dt * z2[:, k] - 0.5 * self.nu**2 * dt)
                a_int = 0.5 * dt * (a * a + a_next * a_next)  # trapezoidal integrated variance
                # ∫α dW2 = (α' − α)/ν exactly; for ν→0 the vol is constant
                # (α'=α) and the correlation drift vanishes.
                corr = 0.0 if nu_zero else (self.rho / self.nu) * (a_next - a)
                log_f = (
                    log_f
                    - 0.5 * a_int
                    + corr
                    + rho_hat * np.sqrt(a_int) * z_perp[:, k]
                )
                a = a_next
            return np.exp(log_f) - self.shift

        raise ValidationError(f"unknown SABR MC scheme: {scheme}")
