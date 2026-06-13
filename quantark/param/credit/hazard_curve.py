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

import numpy as np

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

    The intensity is floored at zero so a downward shock cannot produce a
    negative hazard rate.
    """

    # Integration resolution (steps per year) for the floored downward shift.
    _STEPS_PER_YEAR = 365

    def __init__(self, base_curve: HazardCurve, shift: float):
        if base_curve is None:
            raise ValidationError("base_curve is required")
        self.base_curve = base_curve
        self.shift = shift
        # Lazily-built cumulative integral of the floored intensity, used only on
        # the term-structured downward-shift path. The grid is extended in place
        # as larger times are requested, so a full pricing sweep (which queries
        # survival at every daily integration point) stays linear rather than
        # re-integrating from zero on every call.
        self._int_times = [0.0]
        self._int_cum = [0.0]
        self._last_intensity = None

    def get_hazard_rate(self, time: float) -> float:
        return max(0.0, self.base_curve.get_hazard_rate(time) + self.shift)

    def get_survival_probability(self, time: float) -> float:
        if time < 0:
            raise ValidationError(f"Time must be non-negative, got {time}")
        if time == 0:
            return 1.0
        if isinstance(self.base_curve, FlatHazardCurve):
            # Flat base: the floored intensity max(0, lambda + shift) is constant
            # over the whole interval, so the survival law stays exact and O(1).
            floored = max(0.0, self.base_curve.hazard_rate + self.shift)
            return math.exp(-floored * time)
        if self.shift >= 0:
            # An upward shift never breaches the floor (base intensity >= 0), so
            # the shifted integral is exactly the base integral + shift * t.
            return self.base_curve.get_survival_probability(time) * math.exp(
                -self.shift * time
            )
        # Term-structured base with a downward shift: the floor may bind on only
        # part of the interval, so integrate max(0, lambda(u) + shift) over the
        # whole [0, time] interval (cached/incremental) rather than assuming the
        # endpoint intensity held throughout.
        return math.exp(-self._floored_integral(time))

    def _floored_integral(self, time: float) -> float:
        """Cumulative integral of ``max(0, lambda(u) + shift)`` over ``[0, time]``."""
        dt = 1.0 / self._STEPS_PER_YEAR
        if self._last_intensity is None:
            self._last_intensity = max(
                0.0, self.base_curve.get_hazard_rate(0.0) + self.shift
            )
        while self._int_times[-1] < time - 1e-12:
            t_prev = self._int_times[-1]
            t_next = min(t_prev + dt, time)
            next_intensity = max(
                0.0, self.base_curve.get_hazard_rate(t_next) + self.shift
            )
            self._int_cum.append(
                self._int_cum[-1]
                + 0.5 * (self._last_intensity + next_intensity) * (t_next - t_prev)
            )
            self._int_times.append(t_next)
            self._last_intensity = next_intensity
        return float(np.interp(time, self._int_times, self._int_cum))

    def __repr__(self) -> str:
        return (
            f"ParallelShiftHazardCurve(shift={self.shift:.4%}, "
            f"base={self.base_curve!r})"
        )
