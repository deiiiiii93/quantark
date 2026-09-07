"""Wrap a trading-time-quoted implied surface for calendar-axis engines.

Spec: docs/superpowers/specs/2026-09-01-trading-clock-vol-design.md §4.2:
get_vol(K, tau_cal) = sigma_inner(K, tau_td) * sqrt(tau_td/tau_cal)
preserves total variance exactly: w_cal(tau) = w_td(to_trading(tau)).
``total_variance`` carries the FULL get_vol query signature so smile
surfaces reproduce the identical inner query, and computes w once from
tau_td so holiday plateaus difference to Dw == 0.0 exactly downstream.

Contract: engine times must be ACT/365 calendar fractions of real dates,
and the map's anchor_date must equal the environment's valuation_date.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantark.param.vol.vol_surface import BlackImpliedVolSurface
from quantark.util.calendar.trading_clock import BusinessTimeMap
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class TradingClockVolSurface(BlackImpliedVolSurface):
    inner: BlackImpliedVolSurface
    time_map: BusinessTimeMap
    is_smile: bool = field(init=False, default=False)

    #: Explicit opt-in for the exact total-variance protocol in
    #: TermCoefficients.from_env — the marker (not the method name alone)
    #: gates the fast path, because ``total_variance`` also exists on
    #: SVIVolSurface with a different (strike, t) arity.
    exposes_exact_total_variance = True

    def __post_init__(self) -> None:
        if not isinstance(self.inner, BlackImpliedVolSurface):
            raise ValidationError("inner must be a BlackImpliedVolSurface")
        object.__setattr__(self, "is_smile", bool(self.inner.is_smile))

    def total_variance(self, strike: float, time_to_maturity, spot: float):
        """w_cal(tau_cal) = sigma_inner(K, tau_td)^2 * tau_td, vectorized."""
        tau_td = self.time_map.to_trading(time_to_maturity)
        arr = np.atleast_1d(np.asarray(tau_td, dtype=float))
        out = np.empty_like(arr)
        for j, u in enumerate(arr):
            if u <= 0.0:
                out[j] = 0.0
            else:
                v = float(self.inner.get_vol(float(strike), float(u), float(spot)))
                out[j] = v * v * u
        if np.ndim(tau_td) == 0:
            return float(out[0])
        return out

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        tau_cal = float(time_to_maturity)
        if tau_cal <= 0.0:
            slope = self.time_map.initial_slope()
            u = 1.0 / (365.0 * 10.0)  # one-tenth day in trading units for the inner lookup
            v = float(self.inner.get_vol(float(strike), u, float(spot)))
            return v * float(np.sqrt(slope))
        tau_td = float(self.time_map.to_trading(tau_cal))
        if tau_td <= 0.0:
            return 0.0  # pure-holiday horizon: zero accrued variance (documented)
        v = float(self.inner.get_vol(float(strike), tau_td, float(spot)))
        return v * float(np.sqrt(tau_td / tau_cal))

    def parallel_shifted(self, shift: float) -> "TradingClockVolSurface":
        """Shift the TRADING-quoted inner and re-expose it through the same map.

        The bump unit is one point of sigma_td (desk decision 2026-09-07,
        patch spec 2026-09-03 §14). The calendar-axis vol this surface returns
        then moves by ``shift * sqrt(tau_td/tau_cal)``, which is tenor-dependent:
        a constant shift of sigma_cal is not a parallel shift of anything the
        surface stores, and it would accrue ``2*shift*sqrt(w*tau_cal)`` of
        variance across a holiday, where the clock's defining property is that
        total variance is flat. The rate and dividend wrappers shift their
        (calendar-quoted) inners for the same reason.
        """
        return TradingClockVolSurface(self.inner.parallel_shifted(shift), self.time_map)

    def to_inner_time(self, time_to_maturity: float) -> float:
        """A time on THIS surface's axis (calendar) expressed on the inner's
        axis (trading). Lets a caller read the inner at one shared coordinate
        instead of letting each wrapper's own anchor pick a different point."""
        return float(self.time_map.to_trading(float(time_to_maturity)))

    def with_time_map(self, time_map: BusinessTimeMap) -> "TradingClockVolSurface":
        """The same inner surface re-expressed through another map (e.g. re-anchored)."""
        return TradingClockVolSurface(self.inner, time_map)

    def __repr__(self) -> str:
        return f"TradingClockVolSurface({self.inner!r}, {self.time_map!r})"
