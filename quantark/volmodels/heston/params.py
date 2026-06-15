"""Heston model parameters."""

from __future__ import annotations

from dataclasses import dataclass

from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class HestonParams:
    """Heston stochastic-volatility parameters.

    Attributes:
        v0: initial variance (>= 0).
        kappa: mean-reversion speed (>= 0).
        theta: long-run variance (>= 0).
        sigma: vol-of-vol (>= 0).
        rho: spot/variance correlation in [-1, 1].
    """

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def __post_init__(self) -> None:
        import math as _math
        for name, val in (("v0", self.v0), ("kappa", self.kappa), ("theta", self.theta),
                          ("sigma", self.sigma), ("rho", self.rho)):
            if not _math.isfinite(val):
                raise ValidationError(f"{name} must be finite, got {val}")
        if self.v0 < 0:
            raise ValidationError(f"v0 must be non-negative, got {self.v0}")
        if self.kappa < 0:
            raise ValidationError(f"kappa must be non-negative, got {self.kappa}")
        if self.theta < 0:
            raise ValidationError(f"theta must be non-negative, got {self.theta}")
        if self.sigma < 0:
            raise ValidationError(f"sigma must be non-negative, got {self.sigma}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValidationError(f"rho must be in [-1, 1], got {self.rho}")

    def feller_satisfied(self) -> bool:
        """True when 2 kappa theta >= sigma^2 (variance stays away from zero)."""
        return 2.0 * self.kappa * self.theta >= self.sigma * self.sigma
