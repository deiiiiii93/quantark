"""
Affine Jump-Diffusion (AJD / CIR-with-jumps) stochastic hazard-rate curve.

The default intensity follows a mean-reverting CIR process with exponentially
distributed upward jumps::

    d lambda_t = kappa (theta - lambda_t) dt + sigma sqrt(lambda_t) dW_t + dJ_t

where the jump arrival intensity is ``gamma`` and the mean jump size is
``mu_jump``. The model is affine, so the survival probability has the
exponential form::

    S(t) = E[exp(-integral_0^t lambda_u du)] = exp(alpha(t) + beta(t) * lambda0)

with ``alpha, beta`` solving the Riccati ODEs below. Presenting AJD as a
:class:`HazardCurve` lets the same reduced-form CDS engine price it — the
pricing algorithm is independent of how survival is generated.
"""
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from scipy.integrate import odeint

from quantark.util.exceptions import ValidationError
from .hazard_curve import HazardCurve


@dataclass
class AJDHazardCurve(HazardCurve):
    """
    Stochastic hazard-rate curve under an affine jump-diffusion model.

    Attributes:
        lambda0: Initial hazard rate (default intensity at t=0), >= 0.
        kappa: Mean-reversion speed.
        theta: Long-run mean intensity, >= 0.
        sigma: Diffusion volatility, >= 0.
        gamma: Jump arrival intensity, >= 0.
        mu_jump: Mean (exponential) jump size, >= 0.
    """

    lambda0: float
    kappa: float = 0.5
    theta: float = 0.02
    sigma: float = 0.1
    gamma: float = 0.0
    mu_jump: float = 0.02
    _cache: Dict[float, float] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.lambda0 < 0:
            raise ValidationError(f"lambda0 must be non-negative, got {self.lambda0}")
        if self.theta < 0:
            raise ValidationError(f"theta must be non-negative, got {self.theta}")
        if self.sigma < 0:
            raise ValidationError(f"sigma must be non-negative, got {self.sigma}")
        if self.gamma < 0:
            raise ValidationError(f"gamma must be non-negative, got {self.gamma}")

    def _riccati(self, y: np.ndarray, t: float) -> np.ndarray:
        """ODEs for alpha(t), beta(t) of the exponential-affine survival law."""
        alpha, beta = y
        safe_denom = max(1.0 - self.mu_jump * beta, 1e-10)
        dalpha_dt = self.kappa * self.theta * beta + self.gamma * (
            1.0 / safe_denom - 1.0
        )
        dbeta_dt = -self.kappa * beta + 0.5 * self.sigma**2 * beta**2 - 1.0
        return np.array([dalpha_dt, dbeta_dt])

    def get_survival_probability(self, time: float) -> float:
        if time < 0:
            raise ValidationError(f"Time must be non-negative, got {time}")
        if time == 0:
            return 1.0
        if time in self._cache:
            return self._cache[time]
        times = np.linspace(0.0, time, max(100, int(time * 100)))
        solution = odeint(self._riccati, [0.0, 0.0], times)
        alpha_t, beta_t = solution[-1, 0], solution[-1, 1]
        survival = float(np.exp(alpha_t + beta_t * self.lambda0))
        survival = min(1.0, max(0.0, survival))
        self._cache[time] = survival
        return survival

    def get_default_density(self, time: float) -> float:
        """Unconditional default density q(t) = -dS/dt (central difference)."""
        if time <= 0:
            return 0.0
        dt = min(0.01, time / 10.0)
        before = self.get_survival_probability(max(0.0, time - dt))
        after = self.get_survival_probability(time + dt)
        return max(0.0, (before - after) / (2.0 * dt))

    def get_hazard_rate(self, time: float) -> float:
        """Effective forward hazard h(t) = q(t) / S(t)."""
        survival = self.get_survival_probability(time)
        if survival <= 0:
            return 0.0
        return self.get_default_density(time) / survival
