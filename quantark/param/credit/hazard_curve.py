"""
Credit hazard-rate (default-intensity) curves.

Reduced-form credit modelling represents default as the first jump of a
Poisson process with intensity (hazard rate) ``lambda(t)``. The survival
probability to time ``t`` is::

    S(t) = exp(-integral_0^t lambda(u) du)

and the (unconditional) default probability density is::

    q(t) = lambda(t) * S(t)

These curves carry no discounting; combine them with a
:class:`~quantark.param.rrf.rate_curve.RateCurve` for present values.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

from quantark.util.exceptions import ValidationError


class HazardCurve(ABC):
    """Abstract base class for credit hazard-rate curves."""

    @abstractmethod
    def get_hazard_rate(self, time: float) -> float:
        """Instantaneous hazard rate (default intensity) ``lambda(t)``."""

    @abstractmethod
    def get_survival_probability(self, time: float) -> float:
        """Probability of survival to ``time``: ``S(t)``."""

    def get_default_probability(self, time: float) -> float:
        """Cumulative default probability by ``time``: ``1 - S(t)``."""
        return 1.0 - self.get_survival_probability(time)

    def get_default_density(self, time: float) -> float:
        """Unconditional default probability density ``q(t) = lambda(t) S(t)``."""
        return self.get_hazard_rate(time) * self.get_survival_probability(time)


@dataclass
class FlatHazardCurve(HazardCurve):
    """
    Flat (constant) hazard-rate curve.

    A constant default intensity ``lambda`` gives an exponential survival
    law ``S(t) = exp(-lambda t)``.

    Attributes:
        hazard_rate: Constant annualized hazard rate (default intensity), >= 0.
    """

    hazard_rate: float

    def __post_init__(self) -> None:
        if self.hazard_rate < 0:
            raise ValidationError(
                f"Hazard rate must be non-negative, got {self.hazard_rate}"
            )

    def get_hazard_rate(self, time: float) -> float:
        return self.hazard_rate

    def get_survival_probability(self, time: float) -> float:
        if time < 0:
            raise ValidationError(f"Time must be non-negative, got {time}")
        return math.exp(-self.hazard_rate * time)

    def __repr__(self) -> str:
        return f"FlatHazardCurve(hazard_rate={self.hazard_rate:.4%})"


class ParallelShiftHazardCurve(HazardCurve):
    """
    Parallel-shift wrapper applying a constant additive shift to a base
    hazard curve's intensity. Used for CS01 / spread-shock revaluation::

        lambda_shift(t) = max(0, lambda_base(t) + shift)
        S_shift(t)      = exp(-integral lambda_shift)

    For a flat base curve this is exact. The intensity is floored at zero so a
    downward shock cannot produce a negative hazard rate.
    """

    def __init__(self, base_curve: HazardCurve, shift: float):
        if base_curve is None:
            raise ValidationError("base_curve is required")
        self.base_curve = base_curve
        self.shift = shift

    def get_hazard_rate(self, time: float) -> float:
        return max(0.0, self.base_curve.get_hazard_rate(time) + self.shift)

    def get_survival_probability(self, time: float) -> float:
        if time < 0:
            raise ValidationError(f"Time must be non-negative, got {time}")
        base = self.base_curve.get_survival_probability(time)
        # Floored shift: when the base intensity stays above -shift everywhere,
        # the integral of the shifted intensity is the base integral + shift*t.
        floor_breached = self.base_curve.get_hazard_rate(time) + self.shift < 0
        if floor_breached:
            return math.exp(-self.get_hazard_rate(time) * time)
        return base * math.exp(-self.shift * time)

    def __repr__(self) -> str:
        return (
            f"ParallelShiftHazardCurve(shift={self.shift:.4%}, "
            f"base={self.base_curve!r})"
        )
