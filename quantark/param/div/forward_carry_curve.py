"""Cumulative log-carry curve B(T) = log(F(0,T)/S0) (spec WP3.1).

Conventions (normative): B interpolated piecewise-linear between nodes,
B(0) = 0; beyond the last node per the carry extrapolation scheme
(default FLAT_FORWARD_CARRY: the last segment's slope dB/dt continues;
ZERO_FORWARD_CARRY holds B flat). q(T)*T = r(T)*T - B(T) node-exactly via
``to_dividend_yield``.

Integration route: the ``to_dividend_yield`` adapter feeds the existing
``PricingEnvironment.div_yield`` slot, so ``forward_carry_on_grid`` keeps
working unchanged (documented adapter route per spec WP3.1).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError


class ForwardCarryCurve:
    def __init__(
        self,
        nodes: List[Tuple[float, float]],
        node_roles=None,
        last_observable_tenor: Optional[float] = None,
        extrapolation=None,
    ):
        from quantark.param.extrapolation import CarryExtrapolation

        if not nodes:
            raise ValidationError("ForwardCarryCurve requires at least one node")
        t = np.array([float(n[0]) for n in nodes])
        b = np.array([float(n[1]) for n in nodes])
        if np.any(t <= 0.0):
            raise ValidationError("node tenors must be positive (B(0)=0 implicit)")
        if t.size > 1 and np.any(np.diff(t) <= 0.0):
            raise ValidationError("node tenors must be strictly increasing")
        if not np.all(np.isfinite(b)):
            raise ValidationError("carry values must be finite")
        self._t = np.concatenate([[0.0], t])
        self._b = np.concatenate([[0.0], b])
        self.node_roles = node_roles
        self.last_observable_tenor = (
            float(last_observable_tenor)
            if last_observable_tenor is not None
            else float(t[-1])
        )
        self.extrapolation = (
            extrapolation
            if extrapolation is not None
            else CarryExtrapolation.FLAT_FORWARD_CARRY
        )

    @property
    def nodes(self) -> List[Tuple[float, float]]:
        return list(zip(self._t[1:].tolist(), self._b[1:].tolist()))

    def carry(self, T) -> float:
        from quantark.param.extrapolation import CarryExtrapolation

        T = float(T)
        if T < 0.0:
            raise ValidationError(f"T must be >= 0, got {T}")
        if T <= self._t[-1]:
            return float(np.interp(T, self._t, self._b))
        if self.extrapolation is CarryExtrapolation.ZERO_FORWARD_CARRY:
            return float(self._b[-1])
        # FLAT_FORWARD_CARRY: last segment slope continues (continuous at
        # the last node by construction)
        slope = (self._b[-1] - self._b[-2]) / (self._t[-1] - self._t[-2])
        return float(self._b[-1] + slope * (T - self._t[-1]))

    def forward(self, s0: float, T: float) -> float:
        return float(s0) * float(np.exp(self.carry(T)))

    def interval_carry(self, t0: float, t1: float) -> float:
        t0, t1 = float(t0), float(t1)
        if t1 < t0:
            raise ValidationError("t1 must be >= t0")
        if t1 == t0:
            # right forward carry dB/dt+ at t0: slope of the containing segment
            idx = int(np.searchsorted(self._t, t0, side="right"))
            idx = min(max(idx, 1), self._t.size - 1)
            return float(
                (self._b[idx] - self._b[idx - 1])
                / (self._t[idx] - self._t[idx - 1])
            )
        return (self.carry(t1) - self.carry(t0)) / (t1 - t0)

    def to_dividend_yield(self, rate_curve):
        """Dividend-yield adapter with q(T)*T = r(T)*T - B(T) POINTWISE.

        Returns a view over this curve and the rate curve (not a resampled
        TermStructureDividendYield): sampling q at nodes and re-interpolating
        q linearly would distort B(T) between and beyond nodes for non-flat
        carry. The view honors this curve's extrapolation scheme and carries
        its node metadata.
        """
        return _CarryImpliedDividendYield(carry_curve=self, rate_curve=rate_curve)

    @classmethod
    def from_index_futures(cls, futures_curve, rate_curve) -> "ForwardCarryCurve":
        """B(T_i) = (r(T_i) - q_i) * T_i from futures-implied yields."""
        qs = futures_curve.implied_yields(rate_curve)
        nodes = [
            (
                float(quote.maturity),
                (float(rate_curve.get_rate(quote.maturity)) - float(q))
                * float(quote.maturity),
            )
            for quote, q in zip(futures_curve.quotes, qs)
        ]
        return cls(nodes)

    @classmethod
    def from_forward_nodes(cls, s0, nodes) -> "ForwardCarryCurve":
        if float(s0) <= 0.0:
            raise ValidationError("s0 must be positive")
        return cls(
            [(float(T), float(np.log(float(F) / float(s0)))) for T, F in nodes]
        )


from quantark.param.div.dividend_yield import DividendYield  # noqa: E402


class _CarryImpliedDividendYield(DividendYield):
    """DividendYield view: q(T) = (r(T)*T - B(T)) / T pointwise.

    Exact for every T (interpolated, node, and extrapolated regions alike);
    array inputs are supported for the grid samplers. Entries at T <= 0 map
    to 0.0 in ARRAY calls only — grid samplers weight by q(T)*T, so the
    T = 0 entry is mathematically irrelevant; scalar calls at T <= 0 raise.
    """

    def __init__(self, carry_curve: ForwardCarryCurve, rate_curve):
        self._carry = carry_curve
        self._rate = rate_curve
        self.node_roles = carry_curve.node_roles
        self.last_observable_tenor = carry_curve.last_observable_tenor

    def get_yield(self, time_to_maturity):
        t = time_to_maturity
        if np.ndim(t) == 0:
            t = float(t)
            if t <= 0.0:
                raise ValidationError(
                    f"carry-implied yield needs T > 0, got {t}"
                )
            return (
                float(self._rate.get_rate(t)) * t - self._carry.carry(t)
            ) / t
        arr = np.asarray(t, dtype=float)
        out = np.zeros_like(arr)
        pos = arr > 0.0
        for idx in np.flatnonzero(pos):
            ti = float(arr.flat[idx])
            out.flat[idx] = (
                float(self._rate.get_rate(ti)) * ti - self._carry.carry(ti)
            ) / ti
        return out
