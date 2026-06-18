"""Rate-curve helpers for term-structure-consistent IR exposure (#3).

Two pieces:

- ``ForwardRateCurve`` presents a base curve *as seen from* a future time ``t0`` (the
  roll-down forward curve): ``DF(tau) = DF_base(t0+tau)/DF_base(t0)``. Used by the as-of
  repricer so a trade valued at a future node sees the correct forward rates, not the
  original curve re-anchored.

- ``key_rate_bumped_curve`` shifts ONE pillar's zero rate of an interpolated curve by an
  absolute amount, holding the others — the SA-CVA IR delta key-rate bump. Flat curves
  have no per-tenor pillars, so they are rejected (a flat curve cannot produce regulatory
  per-tenor factors — exactly the documented IR-delta precondition).
"""
from __future__ import annotations

import math
from typing import List

from quantark.param.rrf.rate_curve import InterpolatedRateCurve, RateCurve
from quantark.util.exceptions import ValidationError

# SA-CVA interest-rate delta vertices (MAR50.55, reduced set).
IR_DELTA_TENORS = (1.0, 2.0, 5.0, 10.0, 30.0)


class ForwardRateCurve(RateCurve):
    """Base curve seen from time ``t0`` (year-fraction from the base valuation date).

    ``DF(tau) = DF_base(t0 + tau) / DF_base(t0)``; the zero rate is derived from it. For a
    flat base curve this is numerically identical to the base curve, so wrapping is safe.
    """

    def __init__(self, base_curve: RateCurve, t0: float):
        if base_curve is None:
            raise ValidationError("base_curve is required")
        if not (t0 >= 0) or not math.isfinite(t0):
            raise ValidationError(f"t0 must be finite and >= 0, got {t0}")
        self.base_curve = base_curve
        self.t0 = float(t0)
        self._df0 = float(base_curve.get_discount_factor(self.t0))
        if not (self._df0 > 0):
            raise ValidationError("base discount factor at t0 must be positive")

    def get_discount_factor(self, time_to_maturity: float) -> float:
        if time_to_maturity < 0:
            raise ValidationError(
                f"Time to maturity must be non-negative, got {time_to_maturity}")
        df = float(self.base_curve.get_discount_factor(self.t0 + time_to_maturity))
        return df / self._df0

    def get_rate(self, time_to_maturity: float) -> float:
        # The forward zero rate is z(tau) = -ln(DF(tau))/tau for tau>0; its tau->0 limit is
        # the instantaneous forward f(0,t0) = -d/dT ln P(0,T)|_{t0}, which the base curve
        # does not expose exactly. Repricers query strictly positive tenors, so refuse
        # tau<=0 rather than return a wrong (spot-zero) rate.
        if time_to_maturity <= 0:
            raise ValidationError(
                "ForwardRateCurve.get_rate requires time_to_maturity > 0 (the tau=0 "
                "instantaneous forward is not exactly available from the base curve)")
        return -math.log(self.get_discount_factor(time_to_maturity)) / time_to_maturity

    def __repr__(self):
        return f"ForwardRateCurve(t0={self.t0:g}, base={self.base_curve!r})"


def key_rate_bumped_curve(curve: RateCurve, tenor: float, bump: float) -> RateCurve:
    """Return a copy of ``curve`` with the pillar at ``tenor`` shifted by ``bump`` (an
    absolute zero-rate shift), all other pillars held. The curve must be interpolated
    (have pillars) and contain that tenor exactly — no silent nearest-pillar snapping."""
    if not isinstance(curve, InterpolatedRateCurve):
        raise ValidationError(
            "key-rate IR delta requires an InterpolatedRateCurve (term structure); a flat "
            "curve has no per-tenor pillars to bump")
    pillars: List = list(curve.pillars)
    idx = [i for i, (t, _) in enumerate(pillars) if t == tenor]
    if len(idx) != 1:
        raise ValidationError(
            f"curve has no unique pillar at tenor {tenor}; pillars are "
            f"{[t for t, _ in pillars]}")
    i = idx[0]
    t_i, r_i = pillars[i]
    bumped = list(pillars)
    bumped[i] = (t_i, r_i + bump)
    return type(curve)(bumped)
