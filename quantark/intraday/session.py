"""Exchange session definition for intraday mode.

A market calendar (holidays/weekends, the existing ``Calendar``) plus the
wall-clock sessions of the exchange and its timezone. Date-only contractual
observations resolve to ``close_at(date)``; date-only cashflows to
``payment_at(date)``. Local-time inputs that fall in a DST gap or fold are
rejected explicitly — never guessed (design: Gate A).
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from quantark.execution.cache.fingerprint import fingerprint
from quantark.util.calendar import Calendar
from quantark.util.exceptions import ValidationError

_SEARCH_DAYS = 3660  # ten years: a calendar with no trading day in that span is broken


def calendar_identity(calendar: Calendar) -> tuple:
    """Value identity of the holiday/weekend rules that decide which days accrue variance.

    ``Calendar`` is mutable and its name is free text, so neither the object nor
    its name identifies its contents: two calendars called "SSE" can disagree on
    one holiday and price a contract differently. The holiday set is fingerprinted
    (it is long and its members are opaque); the rest stays readable.
    """
    holidays, weekend = getattr(calendar, "holidays", None), getattr(calendar, "weekend_days", None)
    if holidays is None or weekend is None:
        raise ValidationError(f"{type(calendar).__name__} exposes no holidays/weekend_days: the intraday clock cannot "
                              "identify the variance schedule it would price on")
    # Holidays are normalized to midnight, so a proleptic ordinal is the whole date and is
    # an order of magnitude cheaper to build than an ISO string per holiday -- this runs
    # twice per valuation over a calendar with hundreds of entries.
    return ("calendar", type(calendar).__name__, str(getattr(calendar, "name", "")),
            tuple(sorted(int(w) for w in weekend)),
            fingerprint(tuple(sorted(d.toordinal() for d in holidays))))


@dataclass(frozen=True)
class TradingSession:
    """One continuous trading session in local wall-clock time (naive ``time`` objects)."""

    open: time
    close: time

    def __post_init__(self):
        if not isinstance(self.open, time) or not isinstance(self.close, time):
            raise ValidationError("session open/close must be datetime.time")
        if self.open.tzinfo is not None or self.close.tzinfo is not None:
            raise ValidationError("session times are local wall-clock: pass naive time objects")
        if self.close <= self.open:
            raise ValidationError(f"session close {self.close} must be after open {self.open}")


@dataclass(frozen=True)
class TradingSessionCalendar:
    """Holidays + ordered sessions + timezone of one exchange.

    ``early_closes`` maps a date to the close time of that day's LAST session;
    sessions opening at or after it are dropped. ``payment_time`` is the local
    time a date-only cashflow is deemed paid (default: the normal close).
    """

    name: str
    tz: tzinfo
    calendar: Calendar
    sessions: Tuple[TradingSession, ...]
    early_closes: Mapping[date, time] = field(default_factory=dict)
    payment_time: Optional[time] = None

    def __post_init__(self):
        if not self.name:
            raise ValidationError("session calendar needs a name")
        if not isinstance(self.tz, tzinfo):
            raise ValidationError("tz must be a tzinfo")
        object.__setattr__(self, "sessions", tuple(self.sessions))
        if not self.sessions:
            raise ValidationError("at least one trading session is required")
        for prev, cur in zip(self.sessions, self.sessions[1:]):
            if cur.open < prev.close:
                raise ValidationError(f"sessions overlap or are unordered: {prev} then {cur}")
        object.__setattr__(self, "early_closes", MappingProxyType(dict(self.early_closes)))
        for d, t in self.early_closes.items():
            if t <= self.sessions[0].open:
                raise ValidationError(f"early close {t} on {d} is not after the first open")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        """Immutable value object: copies share it (engines deep-copy their environments)."""
        return self

    # --- trading days -------------------------------------------------
    def is_trading_day(self, d: date) -> bool:
        return bool(self.calendar.is_business_day(datetime(d.year, d.month, d.day)))

    def next_trading_day(self, d: date) -> date:
        cur = d + timedelta(days=1)
        for _ in range(_SEARCH_DAYS):
            if self.is_trading_day(cur):
                return cur
            cur += timedelta(days=1)
        raise ValidationError(f"no trading day within 10 years after {d}")

    def previous_trading_day(self, d: date) -> date:
        cur = d - timedelta(days=1)
        for _ in range(_SEARCH_DAYS):
            if self.is_trading_day(cur):
                return cur
            cur -= timedelta(days=1)
        raise ValidationError(f"no trading day within 10 years before {d}")

    # --- localization -------------------------------------------------
    def localize(self, d: date, t: time) -> datetime:
        """Aware instant of local wall-clock ``t`` on ``d``; DST gaps and folds raise."""
        naive = datetime.combine(d, t)
        a = naive.replace(tzinfo=self.tz, fold=0)
        # A gap is checked first: zoneinfo also reports different fold offsets
        # inside a gap, which would otherwise be misread as a fold.
        back = a.astimezone(timezone.utc).astimezone(self.tz)
        if back.replace(tzinfo=None) != naive:
            raise ValidationError(f"local time {naive} is nonexistent in {self.tz} (DST gap); supply an explicit instant")
        b = naive.replace(tzinfo=self.tz, fold=1)
        if a.utcoffset() != b.utcoffset():
            raise ValidationError(f"local time {naive} is ambiguous in {self.tz} (DST fold); supply an explicit instant")
        return a

    # --- session geometry ---------------------------------------------
    def session_bounds(self, d: date) -> Tuple[Tuple[datetime, datetime], ...]:
        """(open, close) instants of each session held on ``d``, early close applied."""
        if not self.is_trading_day(d):
            raise ValidationError(f"{d} is not a trading day on {self.name}")
        early = self.early_closes.get(d)
        out = []
        for s in self.sessions:
            close = s.close
            if early is not None:
                if early <= s.open:
                    break
                close = min(close, early)
            out.append((self.localize(d, s.open), self.localize(d, close)))
            if early is not None and close == early:
                break
        return tuple(out)

    def open_at(self, d: date) -> datetime:
        return self.session_bounds(d)[0][0]

    def close_at(self, d: date) -> datetime:
        return self.session_bounds(d)[-1][1]

    def payment_at(self, d: date) -> datetime:
        """Deemed payment instant of a date-only cashflow; ``d`` need not be a trading day."""
        return self.localize(d, self.payment_time or self.sessions[-1].close)

    def identity(self) -> tuple:
        """Everything about this clock that moves a price: sessions, timezone AND the holiday calendar.

        Recomputed every call, because ``calendar`` is mutable and a cached identity of an
        edited calendar is exactly the collision this identity exists to prevent. A
        ``snapshot()`` owns its calendar, so that one caches (see ``_frozen_identity``).
        """
        cached = getattr(self, "_frozen_identity", None)
        if cached is not None:
            return cached
        return self._compute_identity()

    def _compute_identity(self) -> tuple:
        return (
            self.name,
            str(self.tz),
            tuple((s.open.isoformat(), s.close.isoformat()) for s in self.sessions),
            tuple(sorted((d.isoformat(), t.isoformat()) for d, t in self.early_closes.items())),
            None if self.payment_time is None else self.payment_time.isoformat(),
            calendar_identity(self.calendar),
        )

    def snapshot(self) -> "TradingSessionCalendar":
        """A copy owning its calendar contents; the caller may keep editing theirs.

        Only the holiday and weekend SETS are mutable — their members are dates, which are
        not — so a fresh set of the same members isolates the snapshot exactly. Deep-copying
        hundreds of holiday datetimes per valuation would buy nothing. A calendar subclass
        this method does not know is deep-copied instead: it may own other mutable state.
        """
        source = self.calendar
        if type(source) is not Calendar:
            return replace(self, calendar=deepcopy(source))
        clone = copy(source)
        clone.holidays = set(source.holidays)
        clone.weekend_days = set(source.weekend_days)
        snapshot = replace(self, calendar=clone)
        # Nothing may edit a snapshot's calendar, so its identity is fixed from here on and
        # is worth computing once: it hashes every holiday and is read at least twice per
        # valuation (the context identity and the reported session identity).
        object.__setattr__(snapshot, "_frozen_identity", snapshot._compute_identity())
        return snapshot
