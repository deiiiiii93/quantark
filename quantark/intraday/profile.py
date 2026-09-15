"""Variance profile and the intraday variance/carry clock map.

The profile says how one trading day's variance budget (1/D in trading-time
units, D = ``days_per_year``) is split across the segments of the
close-to-close window that ENDS at that day's close: the overnight segment
(previous trading close -> open, weekends and holidays fold into it), each
session, and each intra-day break. Weights are explicit desk inputs and must
sum to one; a zero weight is an exactly-zero-variance segment.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from math import fsum, isfinite
from typing import List, Tuple

import numpy as np

from quantark.intraday.session import TradingSessionCalendar
from quantark.intraday.timestamp import calendar_year_fraction, require_aware, seconds_between, to_utc
from quantark.util.calendar.trading_clock import TradingClock
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

_NORMALIZATION_TOL = 1e-12


class SegmentKind(Enum):
    OVERNIGHT = "overnight"
    SESSION = "session"
    BREAK = "break"


@dataclass(frozen=True)
class VarianceProfile:
    """Versioned split of one trading day's variance budget across its segments.

    ``days_per_year`` is the annualization basis D of the INNER (trading-quoted)
    vol surface. Weights: ``overnight_weight`` (previous close -> open),
    ``session_weights`` (one per session), ``break_weights`` (one per gap
    between consecutive sessions); all finite, non-negative, summing to one.
    """

    name: str
    version: str
    days_per_year: int
    overnight_weight: float
    session_weights: Tuple[float, ...]
    break_weights: Tuple[float, ...]
    holiday_policy: str = "fold_into_overnight"
    shortened_session_policy: str = "same_budget"

    def __post_init__(self):
        if not self.name or not self.version:
            raise ValidationError("VarianceProfile needs a non-empty name and version")
        if int(self.days_per_year) <= 0:
            raise ValidationError("days_per_year must be positive")
        object.__setattr__(self, "days_per_year", int(self.days_per_year))
        object.__setattr__(self, "session_weights", tuple(float(w) for w in self.session_weights))
        object.__setattr__(self, "break_weights", tuple(float(w) for w in self.break_weights))
        object.__setattr__(self, "overnight_weight", float(self.overnight_weight))
        if not self.session_weights:
            raise ValidationError("a profile needs at least one session weight")
        if len(self.break_weights) != len(self.session_weights) - 1:
            raise ValidationError("break_weights must have one entry per gap between sessions")
        weights = (self.overnight_weight, *self.session_weights, *self.break_weights)
        if any((not isfinite(w)) or w < 0.0 for w in weights):
            raise ValidationError("profile weights must be finite and non-negative")
        total = fsum(weights)
        if not is_close(total, 1.0, rel_tol=0.0, abs_tol=_NORMALIZATION_TOL):
            raise ValidationError(f"profile weights must sum to 1 (one trading day's budget), got {total!r}")
        if self.holiday_policy != "fold_into_overnight":
            raise ValidationError(f"unsupported holiday_policy {self.holiday_policy!r}")
        if self.shortened_session_policy != "same_budget":
            raise ValidationError(f"unsupported shortened_session_policy {self.shortened_session_policy!r}")

    def identity(self) -> tuple:
        return (self.name, self.version, self.days_per_year, self.overnight_weight, self.session_weights,
                self.break_weights, self.holiday_policy, self.shortened_session_policy)

    def weights_for(self, n_sessions: int) -> Tuple[Tuple[SegmentKind, float], ...]:
        """Ordered (kind, weight): overnight, session 0, break 0, session 1, ..."""
        if n_sessions != len(self.session_weights):
            raise ValidationError(f"profile declares {len(self.session_weights)} sessions, calendar has {n_sessions}")
        out = [(SegmentKind.OVERNIGHT, self.overnight_weight)]
        for i, w in enumerate(self.session_weights):
            if i > 0:
                out.append((SegmentKind.BREAK, self.break_weights[i - 1]))
            out.append((SegmentKind.SESSION, w))
        return tuple(out)

    @classmethod
    def uniform(cls, session_calendar: TradingSessionCalendar, days_per_year: int,
                reference_date: date, version: str = "1") -> "VarianceProfile":
        """Weights proportional to calendar seconds on ``reference_date`` (constant rate on a normal day)."""
        bounds = session_calendar.session_bounds(reference_date)
        if len(bounds) != len(session_calendar.sessions):
            raise ValidationError(f"{reference_date} is a shortened day; pick a normal reference date")
        prev_close = session_calendar.close_at(session_calendar.previous_trading_day(reference_date))
        overnight = seconds_between(prev_close, bounds[0][0])
        sessions, breaks = [], []
        for i, (o, c) in enumerate(bounds):
            sessions.append(seconds_between(o, c))
            if i > 0:
                breaks.append(seconds_between(bounds[i - 1][1], o))
        total = overnight + sum(sessions) + sum(breaks)
        return cls(name=f"uniform[{session_calendar.name}]", version=version, days_per_year=days_per_year,
                   overnight_weight=overnight / total,
                   session_weights=tuple(s / total for s in sessions),
                   break_weights=tuple(b / total for b in breaks))

    @classmethod
    def sessions_only(cls, session_calendar: TradingSessionCalendar, days_per_year: int,
                      version: str = "1") -> "VarianceProfile":
        """Session weights proportional to session length; overnight and breaks exactly 0.0."""
        anchor = date(2000, 1, 3)
        lengths = [(datetime.combine(anchor, s.close) - datetime.combine(anchor, s.open)).total_seconds()
                   for s in session_calendar.sessions]
        total = sum(lengths)
        return cls(name=f"sessions_only[{session_calendar.name}]", version=version, days_per_year=days_per_year,
                   overnight_weight=0.0, session_weights=tuple(length / total for length in lengths),
                   break_weights=tuple(0.0 for _ in lengths[1:]))


@dataclass(frozen=True)
class Segment:
    """One piece of the variance clock: constant slope, or an exact plateau when ``weight == 0``."""

    start: datetime
    end: datetime
    kind: SegmentKind
    weight: float
    tau_start: float        # calendar year fractions from the anchor
    tau_end: float
    u_start: float          # cumulative trading time (variance clock)
    u_end: float
    trading_day: date       # the day whose budget this segment belongs to


class IntradayTimeMap:
    """BusinessTimeMap-compatible variance/carry clock with sub-day knots.

    ``to_trading`` places variance: slope (w/D)/(segment calendar length)
    inside a segment of weight w, an exact plateau (stored value, no
    arithmetic) inside a zero-weight segment. ``to_calendar`` places carry:
    the continuous piecewise-linear map between the strictly increasing
    (u, tau) knots, plateaus collapsed onto their end — deliberately not the
    pointwise inverse, as in ``BusinessTimeMap``.

    The overnight segment ending at day d's open starts at the PREVIOUS trading
    day's close (weekends and holidays fold in), so a trading day's segments
    sum to exactly 1/D. The first segment is the remainder of the segment that
    contains the anchor, accruing at that segment's full slope.
    """

    def __init__(self, session_calendar: TradingSessionCalendar, profile: VarianceProfile,
                 anchor: datetime, horizon: datetime) -> None:
        require_aware(anchor, "anchor")
        require_aware(horizon, "horizon")
        if to_utc(horizon) <= to_utc(anchor):
            raise ValidationError("horizon must be after anchor")
        self.session_calendar = session_calendar
        self.profile = profile
        self.anchor_date = anchor
        self.horizon_date = horizon
        self.clock = TradingClock(calendar=session_calendar.calendar, days_per_year=profile.days_per_year)
        self.segments = self._build_segments(session_calendar, profile, anchor, horizon)
        self._tau_knots = np.array([self.segments[0].tau_start, *[s.tau_end for s in self.segments]])
        self._u_knots = np.array([self.segments[0].u_start, *[s.u_end for s in self.segments]])
        self._weights = np.array([s.weight for s in self.segments])
        # carry knots: strictly increasing u; a plateau collapses onto its end tau
        u_c, tau_c = [self._u_knots[0]], [self._tau_knots[0]]
        for i in range(1, len(self._u_knots)):
            if self._u_knots[i] > self._u_knots[i - 1]:
                u_c.append(self._u_knots[i])
                tau_c.append(self._tau_knots[i])
            else:
                tau_c[-1] = self._tau_knots[i]
        self._u_carry, self._tau_carry = np.array(u_c), np.array(tau_c)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        """Immutable after construction: copies share it (engines deep-copy their environments)."""
        return self

    # ---- construction --------------------------------------------------
    @staticmethod
    def _day_segments(cal: TradingSessionCalendar, profile: VarianceProfile, d: date) -> List[tuple]:
        """(start, end, kind, weight) for trading day ``d``: overnight, session, break, session, ..."""
        bounds = cal.session_bounds(d)
        prev_close = cal.close_at(cal.previous_trading_day(d))
        weights = profile.weights_for(len(cal.sessions))
        n_present = len(bounds)
        # An early close drops trailing sessions; under "same_budget" their weight
        # (and the breaks before them) moves onto the last session held.
        present = [weights[0][1]]
        for i in range(n_present):
            if i > 0:
                present.append(weights[2 * i][1])      # break before session i
            present.append(weights[2 * i + 1][1])      # session i
        if n_present < len(cal.sessions):
            dropped = fsum(w for _, w in weights[2 * n_present:])
            present[-1] = present[-1] + dropped
        out = [(prev_close, bounds[0][0], SegmentKind.OVERNIGHT, present[0])]
        j = 1
        for i, (o, c) in enumerate(bounds):
            if i > 0:
                out.append((bounds[i - 1][1], o, SegmentKind.BREAK, present[j]))
                j += 1
            out.append((o, c, SegmentKind.SESSION, present[j]))
            j += 1
        return out

    @classmethod
    def _build_segments(cls, cal, profile, anchor, horizon) -> Tuple[Segment, ...]:
        inv_d = 1.0 / float(profile.days_per_year)
        anchor_utc, horizon_utc = to_utc(anchor), to_utc(horizon)
        d = anchor.astimezone(cal.tz).date()
        if not cal.is_trading_day(d) or anchor_utc >= to_utc(cal.close_at(d)):
            d = cal.next_trading_day(d)
        segments, u = [], 0.0
        while True:
            day_segments = cls._day_segments(cal, profile, d)
            for start, end, kind, w in day_segments:
                if to_utc(end) <= anchor_utc:
                    continue
                seg_start = anchor if to_utc(start) < anchor_utc else start
                frac = seconds_between(seg_start, end) / seconds_between(start, end)
                du = w * inv_d * frac
                segments.append(Segment(start=seg_start, end=end, kind=kind, weight=w,
                                        tau_start=calendar_year_fraction(anchor, seg_start),
                                        tau_end=calendar_year_fraction(anchor, end),
                                        u_start=u, u_end=u + du, trading_day=d))
                u = segments[-1].u_end
            if to_utc(day_segments[-1][1]) >= horizon_utc:
                break
            d = cal.next_trading_day(d)
        return tuple(segments)

    # ---- queries -------------------------------------------------------
    def _locate(self, tau: np.ndarray) -> np.ndarray:
        if np.any(tau < 0.0):
            raise ValidationError("IntradayTimeMap: negative calendar time")
        if np.any(tau > self._tau_knots[-1]):
            raise ValidationError(f"IntradayTimeMap: query beyond horizon {self.horizon_date.isoformat()}")
        idx = np.searchsorted(self._tau_knots, tau, side="right") - 1
        return np.clip(idx, 0, len(self.segments) - 1)

    def to_trading(self, tau_cal):
        """Calendar year fraction → trading year fraction (variance clock)."""
        tau = np.asarray(tau_cal, dtype=float)
        scalar = tau.ndim == 0
        tau = np.atleast_1d(tau)
        i = self._locate(tau)
        t0, t1 = self._tau_knots[i], self._tau_knots[i + 1]
        u0, u1 = self._u_knots[i], self._u_knots[i + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(t1 > t0, (tau - t0) / (t1 - t0), 0.0)
        # plateau: the stored knot VALUE, no arithmetic (bitwise-equality contract)
        out = np.where(self._weights[i] == 0.0, u0, u0 + frac * (u1 - u0))
        return float(out[0]) if scalar else out

    def to_calendar(self, tau_td):
        """Trading year fraction → calendar year fraction (carry clock)."""
        u = np.asarray(tau_td, dtype=float)
        scalar = u.ndim == 0
        u = np.atleast_1d(u)
        if np.any(u < 0.0) or np.any(u > self._u_carry[-1]):
            raise ValidationError("IntradayTimeMap.to_calendar: trading time outside the map")
        out = np.interp(u, self._u_carry, self._tau_carry)
        return float(out[0]) if scalar else out

    def initial_slope(self) -> float:
        """d(to_trading)/d(tau_cal) at 0+; exactly 0.0 inside a zero-weight segment."""
        s = self.segments[0]
        if s.weight == 0.0 or s.tau_end == s.tau_start:
            return 0.0
        return (s.u_end - s.u_start) / (s.tau_end - s.tau_start)

    def variance_time_between(self, tau_a: float, tau_b: float) -> float:
        return float(self.to_trading(tau_b)) - float(self.to_trading(tau_a))

    def segment_at(self, tau_cal: float) -> Segment:
        return self.segments[int(self._locate(np.atleast_1d(float(tau_cal)))[0])]

    def re_anchored(self, anchor: datetime) -> "IntradayTimeMap":
        """The same clock and horizon seen from another valuation instant."""
        return IntradayTimeMap(self.session_calendar, self.profile, anchor, self.horizon_date)

    def identity(self) -> tuple:
        return (self.session_calendar.identity(), self.profile.identity(),
                to_utc(self.anchor_date).isoformat(), to_utc(self.horizon_date).isoformat())

    def __repr__(self) -> str:
        return (f"IntradayTimeMap({self.session_calendar.name}, {self.profile.name}@{self.profile.version}, "
                f"anchor={self.anchor_date.isoformat()}, segments={len(self.segments)})")
