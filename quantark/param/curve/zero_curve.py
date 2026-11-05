"""Zero curve with configurable interpolation and long-end extrapolation.

Spec: ``.ship/2026-11-05-curve-construction/spec.md`` §Contracts ``zero_curve.py``.

``ZeroCurve`` is a :class:`~quantark.param.rrf.rate_curve.RateCurve` built from
``(T, z)`` pillars (continuously compounded zero rates on the ACT/365F axis),
one :class:`CurveInterpolation` scheme and one
:class:`~quantark.param.extrapolation.RateExtrapolation` long-end rule. It is
the output type of the bootstrappers and plugs into
``PricingEnvironment.rate_curve`` unchanged. The curve is immutable: edits go
through :meth:`with_pillars` and :meth:`parallel_shifted`, which return new
curves with the same configuration and metadata.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from quantark.param.curve.interpolation import (
    CurveInterpolation,
    TermInterpolator,
    long_end_rule,
)
from quantark.param.extrapolation import RateExtrapolation
from quantark.param.rrf.rate_curve import ParallelShiftRateCurve, RateCurve
from quantark.util.exceptions import ValidationError

ArrayLike = Union[float, np.ndarray]


class ZeroCurve(RateCurve):
    """Pillar zero curve: ``(T_i, z_i)`` + interpolation scheme + long-end rule.

    Args:
        pillars: ``(time_to_maturity, zero_rate)`` pairs, any order; times must be
            ``> 0`` (``DF(0) = 1`` is implicit) and distinct.
        interpolation: scheme for the level ``z(T)`` / cumulative ``z(T)·T``
            (default log-linear in the discount factor).
        extrapolation: behaviour beyond the last pillar (default: the last
            discrete forward continues).
        node_roles: optional per-pillar :class:`~quantark.param.node_roles.NodeRole`
            metadata (informational; aligned with the SORTED pillars).
        last_observable_tenor: optional last market-observable tenor.
        valuation_date: clock anchor of the time axis; ``None`` means a numeric
            (undated) axis. Builders stamp it and refuse to mix anchors.

    Scalar times return ``float``; ``ndarray`` times return an ``ndarray`` of the
    same shape, computed element-wise by the identical formula.
    """

    def __init__(
        self,
        pillars: Sequence[Tuple[float, float]],
        interpolation: CurveInterpolation = CurveInterpolation.LOG_LINEAR_DF,
        extrapolation: RateExtrapolation = RateExtrapolation.FLAT_FORWARD_RATE,
        node_roles=None,
        last_observable_tenor: Optional[float] = None,
        valuation_date: Optional[datetime] = None,
    ) -> None:
        if not isinstance(interpolation, CurveInterpolation):
            raise ValidationError(
                f"interpolation must be a CurveInterpolation, got {interpolation!r}"
            )
        if not isinstance(extrapolation, RateExtrapolation):
            raise ValidationError(
                f"extrapolation must be a RateExtrapolation, got {extrapolation!r}"
            )
        pairs = [(float(t), float(z)) for t, z in pillars]
        if not pairs:
            raise ValidationError("ZeroCurve needs at least one pillar")
        pairs.sort(key=lambda p: p[0])
        times = np.array([p[0] for p in pairs])
        rates = np.array([p[1] for p in pairs])
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(rates)):
            raise ValidationError("pillar times and rates must be finite")
        if np.any(times <= 0.0):
            raise ValidationError(
                "pillar times must be > 0 (DF(0) = 1 is implicit; drop the T = 0 pillar)"
            )
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            dup = times[1:][np.diff(times) <= 0.0][0]
            raise ValidationError(f"duplicate pillar time {dup}")

        self._pillars: Tuple[Tuple[float, float], ...] = tuple(pairs)
        self.interpolation = interpolation
        self.extrapolation = extrapolation
        self.node_roles = node_roles
        self.last_observable_tenor = (
            float(last_observable_tenor) if last_observable_tenor is not None else None
        )
        self.valuation_date = valuation_date
        self._interp = TermInterpolator(
            times, rates, interpolation, long_end_rule(extrapolation)
        )

    # ----------------------------------------------------------------- pillars

    @property
    def pillars(self) -> List[Tuple[float, float]]:
        """Sorted ``(time, zero_rate)`` pillars (a copy)."""
        return list(self._pillars)

    @property
    def tenors(self) -> List[float]:
        """Pillar times (sorted)."""
        return [p[0] for p in self._pillars]

    @property
    def rates(self) -> List[float]:
        """Pillar zero rates (sorted by time)."""
        return [p[1] for p in self._pillars]

    # ----------------------------------------------------------------- RateCurve API

    def get_rate(self, time_to_maturity: ArrayLike) -> ArrayLike:  # type: ignore[override]
        """Continuously compounded zero rate ``z(T)``; ``z(0)`` is the first pillar level."""
        return self._interp.level(time_to_maturity)

    def get_discount_factor(self, time_to_maturity: ArrayLike) -> ArrayLike:  # type: ignore[override]
        """``DF(T) = exp(-z(T)·T)``; ``DF(0) = 1``; ``T < 0`` raises ``ValidationError``."""
        cumulative = self._interp.cumulative(time_to_maturity)
        if isinstance(cumulative, np.ndarray):
            return np.exp(-cumulative)
        return float(np.exp(-cumulative))

    def instantaneous_forward(self, time_to_maturity: ArrayLike) -> ArrayLike:
        """Right-continuous instantaneous forward rate ``d(z·T)/dT``."""
        return self._interp.forward(time_to_maturity)

    # ----------------------------------------------------------------- derived curves

    def with_pillars(self, pillars: Sequence[Tuple[float, float]]) -> "ZeroCurve":
        """A new curve on ``pillars`` with this curve's configuration and metadata."""
        return ZeroCurve(
            pillars,
            interpolation=self.interpolation,
            extrapolation=self.extrapolation,
            node_roles=self.node_roles,
            last_observable_tenor=self.last_observable_tenor,
            valuation_date=self.valuation_date,
        )

    def parallel_shifted(self, shift: float) -> RateCurve:
        """Every zero rate moved by ``shift``: ``DF(T) -> DF(T)·exp(-shift·T)``.

        For the four linear-operator schemes (linear interpolation and natural
        splines reproduce constants and linear functions; both long-end rules
        commute with a level shift) the pillar shift is exact, so a ``ZeroCurve``
        with shifted pillars is returned. Monotone convex clips node forwards
        and is not linear in the pillars, so it returns the exact
        :class:`ParallelShiftRateCurve` wrapper instead.
        """
        if self.interpolation is CurveInterpolation.MONOTONE_CONVEX:
            return ParallelShiftRateCurve(self, shift)
        s = float(shift)
        return self.with_pillars([(t, z + s) for t, z in self._pillars])

    def __repr__(self) -> str:
        return (
            f"ZeroCurve(pillars={len(self._pillars)}, interpolation={self.interpolation.value}, "
            f"extrapolation={self.extrapolation.value})"
        )
