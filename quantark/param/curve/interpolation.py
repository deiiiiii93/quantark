"""Shared term-interpolation kernel for zero-rate and carry curves.

Spec: ``.ship/2026-11-05-curve-construction/spec.md`` §Decisions "Scheme
definitions" and "Long-end rules".

Every curve served here is a *cumulative quantity* ``Y(T)`` with level
``y(T) = Y(T) / T``:

* rates: ``Y(T) = z(T) * T = -log DF(T)``; the level is the continuously
  compounded zero rate and the forward ``dY/dt`` the instantaneous forward rate;
* carry: ``Y(T) = B(T) = log(F(0, T) / S0)``; the level is the zero carry yield
  and the forward the instantaneous forward carry.

Knots ``0 = T_0 < T_1 < ... < T_n`` with ``Y_0 = 0`` (``DF(0) = 1`` is implicit,
so a knot at ``T = 0`` is rejected). Schemes, with the rate-world reading of
the enum names:

* ``LINEAR_ZERO``: level piecewise-linear on ``[T_1, T_n]``; ``y = y_1`` on ``[0, T_1]``.
* ``LOG_LINEAR_DF``: ``Y`` piecewise-linear through ``(0, 0)`` — piecewise-constant forward.
* ``CUBIC_ZERO``: natural cubic spline in the level on ``T_1..T_n``; ``y = y_1`` on ``[0, T_1]``.
* ``LOG_CUBIC_DF``: natural cubic spline in ``Y`` through ``(0, 0)``.
* ``MONOTONE_CONVEX``: Hagan & West (2006) on the discrete forwards, with the
  published positivity step applied only where the bounding discrete forwards
  are strictly positive (so the interpolant is defined for every knot vector a
  bootstrap solver probes, and a flat input of any sign stays flat).

Beyond ``T_n`` a :class:`LongEndRule` applies; every rule is continuous at
``T_n``. ``forward`` is the right derivative ``dY/dt+`` everywhere, so at
``T_n`` it already reports the long-end forward.

Scalar in → ``float`` out; ``ndarray`` in → ``ndarray`` of the same shape,
computed element-wise by the identical formula.
"""
from __future__ import annotations

from enum import Enum
from typing import Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import CubicSpline

from quantark.param.extrapolation import CarryExtrapolation, RateExtrapolation
from quantark.util.exceptions import ValidationError

ArrayLike = Union[float, np.ndarray]


class CurveInterpolation(Enum):
    """Interpolation scheme of a term curve (see module docstring)."""

    LINEAR_ZERO = "linear_zero"
    LOG_LINEAR_DF = "log_linear_df"
    CUBIC_ZERO = "cubic_zero"
    LOG_CUBIC_DF = "log_cubic_df"
    MONOTONE_CONVEX = "monotone_convex"

    def __str__(self) -> str:
        return self.value


class LongEndRule(Enum):
    """Behaviour beyond the last knot, all continuous at ``T_n``.

    HOLD_LEVEL:      ``y(T) = y_n``                       (flat zero rate)
    HOLD_FORWARD:    ``Y(T) = Y_n + fd_n (T - T_n)``      (last discrete forward continues)
    HOLD_CUMULATIVE: ``Y(T) = Y_n``                       (flat cumulative carry)
    """

    HOLD_LEVEL = "hold_level"
    HOLD_FORWARD = "hold_forward"
    HOLD_CUMULATIVE = "hold_cumulative"


def long_end_rule(scheme) -> LongEndRule:
    """Map a public extrapolation enum onto the kernel's long-end rule."""
    if scheme is RateExtrapolation.FLAT_ZERO_RATE:
        return LongEndRule.HOLD_LEVEL
    if scheme is RateExtrapolation.FLAT_FORWARD_RATE:
        return LongEndRule.HOLD_FORWARD
    if scheme is CarryExtrapolation.FLAT_FORWARD_CARRY:
        return LongEndRule.HOLD_FORWARD
    if scheme is CarryExtrapolation.ZERO_FORWARD_CARRY:
        return LongEndRule.HOLD_CUMULATIVE
    if scheme is CarryExtrapolation.FLAT_CARRY_YIELD:
        return LongEndRule.HOLD_LEVEL
    raise ValidationError(f"unknown extrapolation scheme: {scheme!r}")


# --------------------------------------------------------------------------- Hagan–West pieces


def mc_node_forwards(knots: np.ndarray, discrete_forwards: np.ndarray) -> np.ndarray:
    """Hagan–West node forwards ``f_0..f_n`` from knots ``T_0..T_n`` and ``fd_1..fd_n``.

    Interior ``f_i`` is the time-weighted average of the adjacent discrete
    forwards, written as ``fd_i + w (fd_{i+1} - fd_i)`` so equal inputs give
    ``f_i == fd_i`` exactly. The positivity step clips ``f_i`` into
    ``[0, 2 min(fd_i, fd_{i+1})]`` only where both bounding discrete forwards
    are strictly positive (ends: ``fd_1`` / ``fd_n``); elsewhere the node
    forward is left unclipped, which keeps the interval non-empty.
    """
    tk = np.asarray(knots, dtype=float)
    fd = np.asarray(discrete_forwards, dtype=float)
    n = fd.size
    f = np.empty(n + 1)
    if n == 1:
        f[0] = fd[0]
        f[1] = fd[0]
    else:
        for i in range(1, n):
            w = (tk[i] - tk[i - 1]) / (tk[i + 1] - tk[i - 1])
            f[i] = fd[i - 1] + w * (fd[i] - fd[i - 1])
        f[0] = fd[0] - 0.5 * (f[1] - fd[0])
        f[n] = fd[n - 1] - 0.5 * (f[n - 1] - fd[n - 1])
    # Exact "> 0.0" is deliberate: the clip applies only where the bounding
    # discrete forwards are strictly positive (a zero forward keeps the
    # interval non-empty and leaves f_i alone); no tolerance.
    for i in range(1, n):
        if fd[i - 1] > 0.0 and fd[i] > 0.0:
            f[i] = min(max(f[i], 0.0), 2.0 * min(fd[i - 1], fd[i]))
    if fd[0] > 0.0:
        f[0] = min(max(f[0], 0.0), 2.0 * fd[0])
    if fd[n - 1] > 0.0:
        f[n] = min(max(f[n], 0.0), 2.0 * fd[n - 1])
    return f


def mc_region(g0: float, g1: float) -> Tuple[int, float, float]:
    """Region id (0: ``g ≡ 0``, 1–4: Hagan–West (i)–(iv)), ``eta`` and ``A``.

    The conditions partition the ``(g0, g1)`` plane; ``g0 = 0`` and ``g1 = 0``
    fall into region 4 ("same sign or zero") where ``eta`` is 1 resp. 0 and
    the formulas take their limits.
    """
    g0 = float(g0)
    g1 = float(g1)
    # Exact float comparisons are deliberate: the regions partition the plane
    # and only exact zeros / exact boundaries select the limit formulas. Do not
    # add a tolerance here (it would double-count or drop a region).
    if g0 == 0.0 and g1 == 0.0:
        return 0, 0.0, 0.0
    if (g0 > 0.0 and -0.5 * g0 >= g1 >= -2.0 * g0) or (
        g0 < 0.0 and -0.5 * g0 <= g1 <= -2.0 * g0
    ):
        return 1, 0.0, 0.0
    if (g0 < 0.0 and g1 > -2.0 * g0) or (g0 > 0.0 and g1 < -2.0 * g0):
        return 2, (g1 + 2.0 * g0) / (g1 - g0), 0.0
    if (g0 > 0.0 and 0.0 > g1 > -0.5 * g0) or (g0 < 0.0 and 0.0 < g1 < -0.5 * g0):
        return 3, 3.0 * g1 / (g1 - g0), 0.0
    s = g0 + g1
    return 4, g1 / s, -g0 * g1 / s


def _as_output(out: np.ndarray, scalar: bool, shape) -> ArrayLike:
    if scalar:
        return float(out.reshape(-1)[0])
    return out.reshape(shape)


def mc_g(x: ArrayLike, g0: float, g1: float) -> ArrayLike:
    """Hagan–West ``g(x)`` on ``[0, 1]`` for the interval with ``g0 = f_{i-1} - fd_i``, ``g1 = f_i - fd_i``."""
    arr = np.asarray(x, dtype=float)
    scalar = arr.ndim == 0
    xx = arr.reshape(-1)
    region, eta, A = mc_region(g0, g1)
    with np.errstate(divide="ignore", invalid="ignore"):  # np.where evaluates both arms
        if region == 0:
            out = np.zeros_like(xx)
        elif region == 1:
            out = g0 * (1.0 - 4.0 * xx + 3.0 * xx * xx) + g1 * (-2.0 * xx + 3.0 * xx * xx)
        elif region == 2:
            u = (xx - eta) / (1.0 - eta)
            right = g0 + (g1 - g0) * (u * u)
            out = np.where(xx <= eta, g0, right)
        elif region == 3:
            u = (eta - xx) / eta
            left = g1 + (g0 - g1) * (u * u)
            out = np.where(xx < eta, left, g1)
        else:
            if eta > 0.0:
                u = (eta - xx) / eta
                left = A + (g0 - A) * (u * u)
            else:
                left = np.full_like(xx, A)  # never selected: x < 0 is impossible
            if eta < 1.0:
                u = (xx - eta) / (1.0 - eta)
                right = A + (g1 - A) * (u * u)
            else:
                right = np.full_like(xx, g1)  # selected only at x == 1: the limit g(1) = g1
            out = np.where(xx < eta, left, right)
    return _as_output(out, scalar, arr.shape)


def mc_G(x: ArrayLike, g0: float, g1: float) -> ArrayLike:
    """``G(x) = ∫_0^x g``; ``G(1) = 0`` in every region (the interval average is ``fd_i``)."""
    arr = np.asarray(x, dtype=float)
    scalar = arr.ndim == 0
    xx = arr.reshape(-1)
    region, eta, A = mc_region(g0, g1)
    with np.errstate(divide="ignore", invalid="ignore"):  # np.where evaluates both arms
        if region == 0:
            out = np.zeros_like(xx)
        elif region == 1:
            xx3 = xx * xx * xx
            out = g0 * (xx - 2.0 * xx * xx + xx3) + g1 * (-xx * xx + xx3)
        elif region == 2:
            d = xx - eta
            right = g0 * xx + (g1 - g0) * (d * d * d) / (3.0 * (1.0 - eta) * (1.0 - eta))
            out = np.where(xx <= eta, g0 * xx, right)
        elif region == 3:
            e = eta - xx
            left = g1 * xx + (g0 - g1) * (eta * eta * eta - e * e * e) / (3.0 * eta * eta)
            right = g1 * xx + (g0 - g1) * eta / 3.0
            out = np.where(xx < eta, left, right)
        else:
            if eta > 0.0:
                e = eta - xx
                left = A * xx + (g0 - A) * (eta * eta * eta - e * e * e) / (3.0 * eta * eta)
            else:
                left = A * xx  # never selected
            base = A * xx + (g0 - A) * eta / 3.0
            if eta < 1.0:
                d = xx - eta
                right = base + (g1 - A) * (d * d * d) / (3.0 * (1.0 - eta) * (1.0 - eta))
            else:
                right = base  # only x == 1, where (x - eta) == 0
            out = np.where(xx < eta, left, right)
    return _as_output(out, scalar, arr.shape)


# --------------------------------------------------------------------------- the kernel


class TermInterpolator:
    """Interpolate a cumulative term quantity ``Y(T)`` through knots ``(T_i, y_i)``.

    Args:
        times: knot times, strictly increasing, all ``> 0``.
        levels: knot levels ``y_i = Y_i / T_i`` (zero rates or zero carry yields).
        interpolation: the :class:`CurveInterpolation` scheme.
        long_end: the :class:`LongEndRule` beyond the last knot.
    """

    def __init__(
        self,
        times: Sequence[float],
        levels: Sequence[float],
        interpolation: CurveInterpolation,
        long_end: LongEndRule,
    ) -> None:
        if not isinstance(interpolation, CurveInterpolation):
            raise ValidationError(f"interpolation must be a CurveInterpolation, got {interpolation!r}")
        if not isinstance(long_end, LongEndRule):
            raise ValidationError(f"long_end must be a LongEndRule, got {long_end!r}")
        t = np.asarray(list(times), dtype=float).reshape(-1)
        y = np.asarray(list(levels), dtype=float).reshape(-1)
        if t.size == 0:
            raise ValidationError("TermInterpolator needs at least one knot")
        if t.shape != y.shape:
            raise ValidationError(f"times ({t.size}) and levels ({y.size}) differ in length")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
            raise ValidationError("knot times and levels must be finite")
        if np.any(t <= 0.0):
            raise ValidationError("knot times must be > 0 (Y(0) = 0 is implicit)")
        if t.size > 1 and np.any(np.diff(t) <= 0.0):
            raise ValidationError("knot times must be strictly increasing")

        self.interpolation = interpolation
        self.long_end = long_end
        self._t = t
        self._y = y
        self._Y = y * t
        self._tk = np.concatenate([[0.0], t])
        self._Yk = np.concatenate([[0.0], self._Y])
        self._fd = np.diff(self._Yk) / np.diff(self._tk)
        n = t.size

        self._spline = None
        self._slopes = None
        if interpolation is CurveInterpolation.LINEAR_ZERO:
            self._slopes = np.diff(y) / np.diff(t) if n > 1 else np.zeros(0)
        elif interpolation is CurveInterpolation.CUBIC_ZERO:
            if n > 1:
                self._spline = CubicSpline(t, y, bc_type="natural")
        elif interpolation is CurveInterpolation.LOG_CUBIC_DF:
            self._spline = CubicSpline(self._tk, self._Yk, bc_type="natural")
        elif interpolation is CurveInterpolation.MONOTONE_CONVEX:
            f = mc_node_forwards(self._tk, self._fd)
            self._mc_g0 = f[:-1] - self._fd
            self._mc_g1 = f[1:] - self._fd

    # ----------------------------------------------------------------- properties

    @property
    def times(self) -> np.ndarray:
        """Knot times (copy)."""
        return self._t.copy()

    @property
    def levels(self) -> np.ndarray:
        """Knot levels (copy)."""
        return self._y.copy()

    @property
    def discrete_forwards(self) -> np.ndarray:
        """``fd_i = (Y_i - Y_{i-1}) / (T_i - T_{i-1})`` with ``T_0 = 0`` (copy)."""
        return self._fd.copy()

    # ----------------------------------------------------------------- public API

    def cumulative(self, t: ArrayLike) -> ArrayLike:
        """``Y(t)``; ``Y(0) = 0``; ``t < 0`` raises ``ValidationError``."""
        arr, flat, scalar = self._prepare(t)
        out = np.empty_like(flat)
        inside = flat <= self._t[-1]
        if np.any(inside):
            out[inside] = self._cumulative_inside(flat[inside])
        if np.any(~inside):
            out[~inside] = self._cumulative_long(flat[~inside])
        return _as_output(out, scalar, arr.shape)

    def level(self, t: ArrayLike) -> ArrayLike:
        """``y(t) = Y(t) / t`` for ``t > 0``; ``y(0) = y_1`` by convention."""
        arr, flat, scalar = self._prepare(t)
        cum = np.asarray(self.cumulative(flat), dtype=float).reshape(-1)
        positive = flat > 0.0
        safe_t = np.where(positive, flat, 1.0)
        out = np.where(positive, cum / safe_t, self._y[0])
        return _as_output(out, scalar, arr.shape)

    def forward(self, t: ArrayLike) -> ArrayLike:
        """Right derivative ``dY/dt+``: the instantaneous forward (rate or carry)."""
        arr, flat, scalar = self._prepare(t)
        out = np.empty_like(flat)
        inside = flat < self._t[-1]
        if np.any(inside):
            out[inside] = self._forward_inside(flat[inside])
        if np.any(~inside):
            out[~inside] = self._forward_long(flat[~inside])
        return _as_output(out, scalar, arr.shape)

    # ----------------------------------------------------------------- internals

    @staticmethod
    def _prepare(t: ArrayLike):
        arr = np.asarray(t, dtype=float)
        flat = arr.reshape(-1)
        if not np.all(np.isfinite(flat)):
            raise ValidationError("times must be finite")
        if np.any(flat < 0.0):
            raise ValidationError("times must be non-negative")
        return arr, flat, arr.ndim == 0

    def _cumulative_inside(self, t: np.ndarray) -> np.ndarray:
        scheme = self.interpolation
        if scheme is CurveInterpolation.LINEAR_ZERO:
            return np.interp(t, self._t, self._y) * t
        if scheme is CurveInterpolation.LOG_LINEAR_DF:
            return np.interp(t, self._tk, self._Yk)
        if scheme is CurveInterpolation.CUBIC_ZERO:
            spline = self._spline
            if spline is None:
                return self._y[0] * t
            lvl = np.where(t < self._t[0], self._y[0], spline(np.maximum(t, self._t[0])))
            return lvl * t
        if scheme is CurveInterpolation.LOG_CUBIC_DF:
            spline = self._spline
            assert spline is not None
            return np.asarray(spline(t), dtype=float)
        return self._mc_cumulative(t)

    def _forward_inside(self, t: np.ndarray) -> np.ndarray:
        scheme = self.interpolation
        if scheme is CurveInterpolation.LINEAR_ZERO:
            slopes = self._slopes
            assert slopes is not None
            idx = np.searchsorted(self._t, t, side="right")  # knots <= t
            out = np.full_like(t, self._y[0])
            seg = idx >= 1
            if np.any(seg):
                j = idx[seg] - 1
                lvl = self._y[j] + slopes[j] * (t[seg] - self._t[j])
                out[seg] = lvl + t[seg] * slopes[j]
            return out
        if scheme is CurveInterpolation.LOG_LINEAR_DF:
            i = np.searchsorted(self._tk, t, side="right")  # interval i: T_{i-1} <= t < T_i
            return self._fd[i - 1]
        if scheme is CurveInterpolation.CUBIC_ZERO:
            spline = self._spline
            if spline is None:
                return np.full_like(t, self._y[0])
            tt = np.maximum(t, self._t[0])
            inner = spline(tt) + tt * spline(tt, 1)
            return np.where(t < self._t[0], self._y[0], inner)
        if scheme is CurveInterpolation.LOG_CUBIC_DF:
            spline = self._spline
            assert spline is not None
            return np.asarray(spline(t, 1), dtype=float)
        return self._mc_forward(t)

    def _cumulative_long(self, t: np.ndarray) -> np.ndarray:
        tn, yn, Yn, fdn = self._t[-1], self._y[-1], self._Y[-1], self._fd[-1]
        if self.long_end is LongEndRule.HOLD_LEVEL:
            return yn * t
        if self.long_end is LongEndRule.HOLD_FORWARD:
            return Yn + fdn * (t - tn)
        return np.full_like(t, Yn)

    def _forward_long(self, t: np.ndarray) -> np.ndarray:
        if self.long_end is LongEndRule.HOLD_LEVEL:
            return np.full_like(t, self._y[-1])
        if self.long_end is LongEndRule.HOLD_FORWARD:
            return np.full_like(t, self._fd[-1])
        return np.zeros_like(t)

    def _mc_intervals(self, t: np.ndarray) -> np.ndarray:
        n = self._t.size
        i = np.searchsorted(self._tk, t, side="right")  # T_{i-1} <= t < T_i
        return np.clip(i, 1, n)  # t == T_n maps to the last interval (x = 1)

    def _mc_cumulative(self, t: np.ndarray) -> np.ndarray:
        idx = self._mc_intervals(t)
        out = np.empty_like(t)
        for i in np.unique(idx):
            m = idx == i
            h = self._tk[i] - self._tk[i - 1]
            x = (t[m] - self._tk[i - 1]) / h
            G = np.asarray(
                mc_G(x, float(self._mc_g0[i - 1]), float(self._mc_g1[i - 1])), dtype=float
            ).reshape(-1)
            out[m] = self._Yk[i - 1] + h * (self._fd[i - 1] * x + G)
        return out

    def _mc_forward(self, t: np.ndarray) -> np.ndarray:
        idx = self._mc_intervals(t)
        out = np.empty_like(t)
        for i in np.unique(idx):
            m = idx == i
            h = self._tk[i] - self._tk[i - 1]
            x = (t[m] - self._tk[i - 1]) / h
            g = np.asarray(
                mc_g(x, float(self._mc_g0[i - 1]), float(self._mc_g1[i - 1])), dtype=float
            ).reshape(-1)
            out[m] = self._fd[i - 1] + g
        return out

    def __repr__(self) -> str:
        return (
            f"TermInterpolator(knots={self._t.size}, interpolation={self.interpolation.value}, "
            f"long_end={self.long_end.value})"
        )
