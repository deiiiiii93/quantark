"""Carry curve ``B(T) = log(F(0, T) / S0)`` with configurable interpolation.

Spec: ``.ship/2026-11-05-curve-construction/spec.md`` §Contracts ``carry_curve.py``.

``CarryCurve`` generalises :class:`~quantark.param.div.forward_carry_curve.ForwardCarryCurve`
(which is exactly ``CarryCurve(nodes)`` with the defaults ``LOG_LINEAR_DF`` +
``FLAT_FORWARD_CARRY``) to every :class:`CurveInterpolation` scheme except
monotone convex, whose positivity step has no meaning for signed carry. The
shared kernel reads the schemes in carry space: the "zero" is the zero carry
yield ``c(T) = B(T)/T`` and the "cumulative" is ``B(T)`` itself.

:meth:`CarryCurve.to_dividend_yield` returns a live
:class:`~quantark.param.div.dividend_yield.DividendYield` view
``q(T) = r(T) - B(T)/T`` over this curve and a rate curve, pointwise exact for
every ``T`` and array-safe for the grid samplers. The curve is immutable:
edits go through :meth:`with_nodes` and :meth:`shifted_yield`.
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
from quantark.param.div.dividend_yield import DividendYield, ParallelShiftDividendYield
from quantark.param.extrapolation import CarryExtrapolation
from quantark.util.exceptions import ValidationError

ArrayLike = Union[float, np.ndarray]

#: Long-end rules under which shifting every node yield by ``s`` is the exact
#: ``q + s`` everywhere (the tail continues a forward or holds a level, both
#: of which move by ``s``); ``ZERO_FORWARD_CARRY`` holds ``B`` and cannot.
_NODE_SHIFT_EXACT_TAILS = frozenset(
    {CarryExtrapolation.FLAT_FORWARD_CARRY, CarryExtrapolation.FLAT_CARRY_YIELD}
)


class CarryCurve:
    """Cumulative log-carry curve on nodes ``(T_i, B_i)``.

    Args:
        nodes: ``(time, B)`` pairs with ``B = log(F(0, T)/S0)``, any order; times
            ``> 0`` (``B(0) = 0`` is implicit) and distinct.
        interpolation: kernel scheme read in carry space (``LINEAR_ZERO`` = linear
            in the zero carry yield ``B/T``, ``LOG_LINEAR_DF`` = linear in ``B``,
            ``CUBIC_ZERO`` / ``LOG_CUBIC_DF`` their natural-spline analogues).
            ``MONOTONE_CONVEX`` raises ``ValidationError``.
        extrapolation: ``FLAT_FORWARD_CARRY`` (last segment's ``dB/dt`` continues),
            ``ZERO_FORWARD_CARRY`` (``B`` flat beyond the last node) or
            ``FLAT_CARRY_YIELD`` (the zero carry yield ``B/T`` held at ``B_n/T_n``).
        node_roles: optional per-node provenance metadata (aligned with the SORTED nodes).
        last_observable_tenor: last market-observable tenor; defaults to the last node.
        valuation_date: clock anchor of the time axis (``None`` = numeric axis).
    """

    def __init__(
        self,
        nodes: Sequence[Tuple[float, float]],
        interpolation: CurveInterpolation = CurveInterpolation.LOG_LINEAR_DF,
        extrapolation: CarryExtrapolation = CarryExtrapolation.FLAT_FORWARD_CARRY,
        node_roles=None,
        last_observable_tenor: Optional[float] = None,
        valuation_date: Optional[datetime] = None,
    ) -> None:
        if not isinstance(interpolation, CurveInterpolation):
            raise ValidationError(
                f"interpolation must be a CurveInterpolation, got {interpolation!r}"
            )
        if interpolation is CurveInterpolation.MONOTONE_CONVEX:
            raise ValidationError(
                "MONOTONE_CONVEX is not defined for carry curves: its positivity step "
                "assumes positive forwards, and forward carry is signed"
            )
        if not isinstance(extrapolation, CarryExtrapolation):
            raise ValidationError(
                f"extrapolation must be a CarryExtrapolation, got {extrapolation!r}"
            )
        pairs = [(float(t), float(b)) for t, b in nodes]
        if not pairs:
            raise ValidationError("CarryCurve needs at least one node")
        pairs.sort(key=lambda p: p[0])
        times = np.array([p[0] for p in pairs])
        carries = np.array([p[1] for p in pairs])
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(carries)):
            raise ValidationError("node times and carries must be finite")
        if np.any(times <= 0.0):
            raise ValidationError("node times must be > 0 (B(0) = 0 is implicit)")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            dup = times[1:][np.diff(times) <= 0.0][0]
            raise ValidationError(f"duplicate node time {dup}")

        self._nodes: Tuple[Tuple[float, float], ...] = tuple(pairs)
        self.interpolation = interpolation
        self.extrapolation = extrapolation
        self.node_roles = node_roles
        self.last_observable_tenor = (
            float(last_observable_tenor)
            if last_observable_tenor is not None
            else float(times[-1])
        )
        self.valuation_date = valuation_date
        self._interp = TermInterpolator(
            times, carries / times, interpolation, long_end_rule(extrapolation)
        )

    # ----------------------------------------------------------------- nodes

    @property
    def nodes(self) -> List[Tuple[float, float]]:
        """Sorted ``(time, B)`` nodes (a copy)."""
        return list(self._nodes)

    @property
    def tenors(self) -> List[float]:
        """Node times (sorted)."""
        return [n[0] for n in self._nodes]

    @property
    def carries(self) -> List[float]:
        """Node cumulative carries ``B_i`` (sorted by time)."""
        return [n[1] for n in self._nodes]

    # ----------------------------------------------------------------- evaluation

    def carry(self, T: ArrayLike) -> ArrayLike:
        """``B(T)``; ``B(0) = 0``; ``T < 0`` raises ``ValidationError``."""
        return self._interp.cumulative(T)

    def forward(self, s0: float, T: ArrayLike) -> ArrayLike:
        """``F(0, T) = S0 · exp(B(T))``."""
        b = self.carry(T)
        if isinstance(b, np.ndarray):
            return float(s0) * np.exp(b)
        return float(s0) * float(np.exp(b))

    def forward_carry(self, T: ArrayLike) -> ArrayLike:
        """Right-continuous instantaneous forward carry ``dB/dt+``."""
        return self._interp.forward(T)

    def interval_carry(self, t0: float, t1: float) -> float:
        """``(B(t1) - B(t0)) / (t1 - t0)``; for ``t1 == t0`` the forward carry ``dB/dt+(t0)``."""
        t0, t1 = float(t0), float(t1)
        if t1 < t0:
            raise ValidationError("t1 must be >= t0")
        if t1 == t0:
            return float(self._interp.forward(t0))
        return (float(self.carry(t1)) - float(self.carry(t0))) / (t1 - t0)

    # ----------------------------------------------------------------- derived curves

    def with_nodes(self, nodes: Sequence[Tuple[float, float]]) -> "CarryCurve":
        """A new curve on ``nodes`` with this curve's configuration and metadata."""
        return CarryCurve(
            nodes,
            interpolation=self.interpolation,
            extrapolation=self.extrapolation,
            node_roles=self.node_roles,
            last_observable_tenor=self.last_observable_tenor,
            valuation_date=self.valuation_date,
        )

    def shifted_yield(self, shift: float) -> "CarryCurve":
        """Nodes ``(T_i, B_i - shift·T_i)``: the implied yield ``q`` moved by ``+shift``.

        Exact at every ``T`` for the four linear-operator schemes under
        ``FLAT_FORWARD_CARRY`` and ``FLAT_CARRY_YIELD`` (both long-end rules
        reproduce a yield moved by a constant); under ``ZERO_FORWARD_CARRY`` the
        flat tail cannot represent ``q + shift`` beyond the last node (the view
        wraps instead).
        """
        s = float(shift)
        return self.with_nodes([(t, b - s * t) for t, b in self._nodes])

    def _check_clock(self, rate_curve: RateCurve) -> None:
        """Two curves carrying different non-None ``valuation_date`` anchors never combine."""
        other = getattr(rate_curve, "valuation_date", None)
        if self.valuation_date is not None and other is not None and other != self.valuation_date:
            raise ValidationError(
                f"carry curve anchored at {self.valuation_date:%Y-%m-%d} but the rate curve at "
                f"{other:%Y-%m-%d}: one clock per run"
            )

    def to_dividend_yield(self, rate_curve) -> "CarryImpliedDividendYield":
        """Live ``DividendYield`` view ``q(T) = r(T) - B(T)/T`` over this curve and ``rate_curve``."""
        self._check_clock(rate_curve)
        return CarryImpliedDividendYield(self, rate_curve)

    def __repr__(self) -> str:
        return (
            f"CarryCurve(nodes={len(self._nodes)}, interpolation={self.interpolation.value}, "
            f"extrapolation={self.extrapolation.value})"
        )


class CarryImpliedDividendYield(DividendYield):
    """``DividendYield`` view: ``q(T) = (r(T)·T - B(T)) / T`` pointwise.

    Exact for every ``T`` (interpolated, node and extrapolated regions alike).
    Scalar calls at ``T <= 0`` raise ``ValidationError``; in ARRAY calls entries at
    ``T <= 0`` map to ``0.0`` — the grid samplers weight by ``q(T)·T``, so that
    entry is mathematically irrelevant (same convention as the legacy
    ``ForwardCarryCurve`` view).
    """

    def __init__(self, carry_curve: CarryCurve, rate_curve) -> None:
        if not isinstance(carry_curve, CarryCurve):
            raise ValidationError("carry_curve must be a CarryCurve")
        if rate_curve is None:
            raise ValidationError("rate_curve is required")
        self.carry_curve = carry_curve
        self.rate_curve = rate_curve
        self.node_roles = carry_curve.node_roles
        self.last_observable_tenor = carry_curve.last_observable_tenor

    def _rates(self, times: np.ndarray) -> np.ndarray:
        """``r(T)`` for every entry, through the curve's own array path when it has one.

        Only a dispatch failure (a scalar-only ``get_rate`` choking on the array:
        ``TypeError`` / ``ValueError``) falls back to per-element scalar calls;
        any other exception is a genuine error and propagates.
        """
        try:
            out = np.asarray(self.rate_curve.get_rate(times), dtype=float)
        except (TypeError, ValueError):
            out = None
        if out is not None and out.shape == times.shape:
            return out
        return np.array([float(self.rate_curve.get_rate(float(t))) for t in times])

    def get_yield(self, time_to_maturity: ArrayLike) -> ArrayLike:  # type: ignore[override]
        arr = np.asarray(time_to_maturity, dtype=float)
        if arr.ndim == 0:
            t = float(arr)
            if t <= 0.0:
                raise ValidationError(f"carry-implied yield needs T > 0, got {t}")
            r = float(self.rate_curve.get_rate(t))
            return (r * t - float(self.carry_curve.carry(t))) / t
        flat = arr.reshape(-1)
        out = np.zeros_like(flat)
        pos = flat > 0.0
        if np.any(pos):
            tp = flat[pos]
            r = self._rates(tp)
            b = np.asarray(self.carry_curve.carry(tp), dtype=float)
            out[pos] = (r * tp - b) / tp
        return out.reshape(arr.shape)

    def parallel_shifted(self, shift: float) -> DividendYield:
        """``q + shift`` everywhere: a node-shifted view under ``FLAT_FORWARD_CARRY``
        or ``FLAT_CARRY_YIELD`` (exact for the linear-operator schemes), else the
        exact wrapper."""
        if self.carry_curve.extrapolation in _NODE_SHIFT_EXACT_TAILS:
            return CarryImpliedDividendYield(
                self.carry_curve.shifted_yield(shift), self.rate_curve
            )
        return ParallelShiftDividendYield(self, float(shift))

    def __repr__(self) -> str:
        return f"CarryImpliedDividendYield(carry={self.carry_curve!r}, rate={self.rate_curve!r})"
