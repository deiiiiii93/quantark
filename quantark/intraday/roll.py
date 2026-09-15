"""Frozen-market time roll: move the valuation instant, keep the absolute market schedule.

A curve or surface is quoted from its valuation instant. Rolling from ts to
ts + dtau with the market frozen means every absolute date keeps its discount
factor ratio, its cumulative carry and its variance:

    DF'(t)   = DF(t + dtau) / DF(dtau)
    q'(t) t  = q(t + dtau)(t + dtau) - q(dtau) dtau
    w'(u)    = w(u + du) - w(du),  w(u) = sigma_inner(K, u)^2 u,  du = to_trading(dtau)

The clock map is re-anchored by ``resolve_context`` (same profile, same
absolute segments), so W'(tau) = W(tau + dtau) - W(dtau) exactly. A roll never
crosses a contract event: that needs an outcome (``roll_through_events``).
Spot and its quote timestamp are unchanged, so no assumption changes.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from math import log, sqrt
from typing import Optional, Sequence

from quantark.execution.errors import DeterminismViolation
from quantark.intraday.events import ContractEvent, EventPhase
from quantark.intraday.fixings import Fixing
from quantark.intraday.timestamp import SECONDS_PER_YEAR, calendar_year_fraction, require_aware, to_utc
from quantark.param.div.dividend_yield import DividendYield
from quantark.param.rrf.rate_curve import RateCurve
from quantark.param.vol.vol_surface import BlackImpliedVolSurface
from quantark.util.exceptions import NumericalError, ValidationError

#: Width of the forward used for a zero-time query on a shifted curve (one calendar second).
_INSTANT = 1.0 / SECONDS_PER_YEAR
_CALENDAR_ARBITRAGE_TOL = 1e-12


@dataclass(frozen=True)
class ShiftedRateCurve(RateCurve):
    """The same curve seen ``shift`` years later: DF'(t) = DF(t + shift) / DF(shift)."""

    inner: RateCurve
    shift: float

    def get_discount_factor(self, time_to_maturity: float) -> float:
        return float(self.inner.get_discount_factor(float(time_to_maturity) + self.shift)) / self._df_shift()

    def get_rate(self, time_to_maturity: float) -> float:
        t = float(time_to_maturity)
        if t <= 0.0:
            return -log(self.get_discount_factor(_INSTANT)) / _INSTANT      # instantaneous forward at the roll date
        return -log(self.get_discount_factor(t)) / t

    def parallel_shifted(self, shift: float) -> "ShiftedRateCurve":
        # zero rates of the inner moved by s: DF'(t) picks up exp(-s t), a parallel shift of this curve too
        return ShiftedRateCurve(self.inner.parallel_shifted(shift), self.shift)

    def _df_shift(self) -> float:
        return float(self.inner.get_discount_factor(self.shift)) if self.shift > 0.0 else 1.0


@dataclass(frozen=True)
class ShiftedDividendYield(DividendYield):
    """Cumulative-carry preserving: q'(t) t = q(t + shift)(t + shift) - q(shift) shift."""

    inner: DividendYield
    shift: float

    def get_yield(self, time_to_maturity: float) -> float:
        t = float(time_to_maturity)
        if t <= 0.0:
            t = _INSTANT
        return (self._carry(t + self.shift) - self._carry(self.shift)) / t

    def parallel_shifted(self, shift: float) -> "ShiftedDividendYield":
        return ShiftedDividendYield(self.inner.parallel_shifted(shift), self.shift)

    def _carry(self, t: float) -> float:
        return float(self.inner.get_yield(t)) * t if t > 0.0 else 0.0


@dataclass(frozen=True)
class ShiftedTradingVolSurface(BlackImpliedVolSurface):
    """A trading-time-quoted surface seen ``delta_u`` of trading time later; total variance preserving."""

    inner: BlackImpliedVolSurface
    delta_u: float
    is_smile: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_smile", bool(getattr(self.inner, "is_smile", False)))

    @property
    def quad_v2_deterministic_variance(self) -> bool:
        from quantark.param.vol.vol_surface import FlatVolSurface, TermStructureVolSurface
        return type(self.inner) in (FlatVolSurface, TermStructureVolSurface) or \
            getattr(self.inner, "quad_v2_deterministic_variance", False) is True

    def _w(self, strike: float, u: float, spot: float) -> float:
        if u <= 0.0:
            return 0.0
        v = float(self.inner.get_vol(strike, u, spot))
        return v * v * u

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        u = float(time_to_maturity)
        if u <= 0.0:
            u = _INSTANT
        forward = self._w(strike, u + self.delta_u, spot) - self._w(strike, self.delta_u, spot)
        if forward < -_CALENDAR_ARBITRAGE_TOL:
            raise NumericalError(f"negative forward variance {forward!r} after the roll: calendar arbitrage in the inner surface")
        return sqrt(max(forward, 0.0) / u)

    def parallel_shifted(self, shift: float) -> "ShiftedTradingVolSurface":
        return ShiftedTradingVolSurface(self.inner.parallel_shifted(shift), self.delta_u)


def next_event_after(ctx, ts: datetime) -> Optional[ContractEvent]:
    key = to_utc(ts, "ts")
    later = [e for e in ctx.numerical.remaining_events if to_utc(e.timestamp) > key]
    return min(later, key=lambda e: to_utc(e.timestamp)) if later else None


def roll_context(ctx, new_ts: datetime, *, frozen_market: bool = True, phase: Optional[EventPhase] = None):
    """The same contract valued at ``new_ts`` with the market frozen (or re-read, ``frozen_market=False``)."""
    from quantark.intraday.context import resolve_context
    require_aware(new_ts, "new_ts")
    ts = ctx.valuation_timestamp
    if to_utc(new_ts) <= to_utc(ts):
        raise ValidationError("a roll moves the valuation instant forward")
    target = to_utc(new_ts)
    crossed = [e for e in ctx.numerical.remaining_events if to_utc(e.timestamp) < target]
    if crossed:
        first = min(crossed, key=lambda e: to_utc(e.timestamp))
        raise ValidationError(f"the roll to {new_ts.isoformat()} crosses {first.event_id}: an event needs an outcome "
                              "(use roll_through_events)")
    phase = EventPhase.parse(phase) if phase is not None else EventPhase.BEFORE
    at_target = [e for e in ctx.numerical.remaining_events if to_utc(e.timestamp) == target]
    if at_target and phase is EventPhase.AFTER:
        raise ValidationError(f"landing AFTER {at_target[0].event_id} needs its outcome (use roll_through_events)")
    request = ctx.request
    env = frozen_env(ctx, new_ts) if frozen_market else dataclasses.replace(request.pricing_env, valuation_date=new_ts)
    rolled = resolve_context(dataclasses.replace(request, pricing_env=env, event_phase=phase))
    if rolled.reconstruction.assumptions != ctx.reconstruction.assumptions:
        raise DeterminismViolation("a roll must not create or drop fixing assumptions")
    return rolled


def frozen_env(ctx, new_ts: datetime):
    """The request's environment re-quoted from ``new_ts`` with every absolute DF ratio, carry and variance kept."""
    env = ctx.request.pricing_env
    d_tau = calendar_year_fraction(ctx.valuation_timestamp, new_ts)
    d_u = float(ctx.time_map.to_trading(d_tau))
    return dataclasses.replace(
        env, valuation_date=new_ts, rate_curve=ShiftedRateCurve(env.rate_curve, d_tau),
        div_yield=None if env.div_yield is None else ShiftedDividendYield(env.div_yield, d_tau),
        vol_surface=ShiftedTradingVolSurface(env.vol_surface, d_u))


def roll_through_events(engine, request, to_timestamp: datetime, *, outcomes: Sequence[Fixing] = (), session=None):
    """A SCENARIO, not a derivative: value the contract at ``to_timestamp`` on the frozen market after the given
    outcomes of every event in between.

    Every remaining event at or before ``to_timestamp`` (including one at the current instant under BEFORE) needs a
    ``Fixing`` at its instant; the scenario lands AFTER an event at ``to_timestamp`` itself.
    """
    from quantark.intraday.context import resolve_context
    from quantark.intraday.service import value_intraday

    require_aware(to_timestamp, "to_timestamp")
    ctx = resolve_context(request)
    target = to_utc(to_timestamp)
    if target <= to_utc(ctx.valuation_timestamp):
        raise ValidationError("roll_through_events moves the valuation instant forward")
    outcomes = tuple(outcomes)
    given = {to_utc(f.timestamp) for f in outcomes}
    crossed = sorted({to_utc(e.timestamp): e.timestamp for e in ctx.numerical.remaining_events
                      if to_utc(e.timestamp) <= target}.items())
    missing = [stamp for key, stamp in crossed if key not in given]
    if missing:
        raise ValidationError("roll_through_events needs an outcome for every crossed event instant; missing "
                              + ", ".join(m.isoformat() for m in missing))
    phase = EventPhase.AFTER if any(key == target for key, _ in crossed) else EventPhase.BEFORE
    scenario = dataclasses.replace(request, pricing_env=frozen_env(ctx, to_timestamp), fixings=request.fixings + outcomes,
                                   event_phase=phase)
    result = value_intraday(engine, scenario, session=session)
    return dataclasses.replace(result, records=result.records + ("scenario:roll_through_events",))
