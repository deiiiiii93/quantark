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
from typing import Tuple

from quantark.intraday.session import TradingSessionCalendar
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
        overnight = (bounds[0][0] - prev_close).total_seconds()
        sessions, breaks = [], []
        for i, (o, c) in enumerate(bounds):
            sessions.append((c - o).total_seconds())
            if i > 0:
                breaks.append((o - bounds[i - 1][1]).total_seconds())
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
