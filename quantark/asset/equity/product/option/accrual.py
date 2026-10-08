"""Accrual time, separate from observation time (spec ``2026-10-08-accrual-time-design.md``).

An observation is a trading-day quantity: when the market is checked. Accrual is a calendar quantity in the contract's
interest basis: how much the contract owes. Each record carries its own immutable ``AccrualPeriod``, filled once at
construction (``accrual_build``), so ageing, snapping or rebasing a schedule moves observation timing and never cash.
The functions at the bottom of this module are the only place accrual is computed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Optional, Union

from quantark.util.calendar import calculate_day_count_fraction, calculate_year_fraction
from quantark.util.calendar.business_calendar import Calendar
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.exceptions import ValidationError

Instant = Union[float, datetime]

BUSINESS_DAYS_TERMS = "BUSINESS_DAYS accrual needs its calendar and denominator on the contract"


class AccrualKind(Enum):
    """What a period's endpoints are: contract-clock years, calendar dates, or none at all."""

    TIME = "time"
    DATE = "date"
    NONE = "none"


@dataclass(frozen=True)
class AccrualBasis:
    """Every term a year fraction reads (spec §4.7): the day count, and for ``BUSINESS_DAYS`` its calendar and
    denominator, frozen with the contract so no accrual reads the pricing environment."""

    day_count: DayCountConvention
    calendar: Optional[Calendar] = None
    bus_days_in_year: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.day_count, DayCountConvention):
            raise ValidationError(f"accrual day count must be DayCountConvention, got {self.day_count!r}")
        if self.calendar is not None and not isinstance(self.calendar, Calendar):
            raise ValidationError(f"accrual calendar must be Calendar, got {type(self.calendar).__name__}")
        if self.bus_days_in_year is not None:
            value = float(self.bus_days_in_year)
            if not isfinite(value) or value <= 0.0:
                raise ValidationError(f"accrual bus_days_in_year must be positive, got {self.bus_days_in_year!r}")
            object.__setattr__(self, "bus_days_in_year", value)

    @property
    def is_complete(self) -> bool:
        """Whether every term this basis's cumulative fraction reads is present."""
        return self.day_count is not DayCountConvention.BUSINESS_DAYS or (
            self.calendar is not None and self.bus_days_in_year is not None)

    def same_terms(self, other: "AccrualBasis") -> bool:
        """Equal day count, denominator and calendar identity (a calendar's ``==`` is object identity)."""
        def ident(calendar):
            return None if calendar is None else calendar.identity()
        return (self.day_count is other.day_count and self.bus_days_in_year == other.bus_days_in_year
                and ident(self.calendar) == ident(other.calendar))

    def cumulative(self, start: datetime, end: datetime) -> float:
        """The cumulative leg's fraction: today's ``calculate_year_fraction`` call, with this basis's terms."""
        if not self.is_complete:
            raise ValidationError(BUSINESS_DAYS_TERMS)
        bus_days = 252 if self.bus_days_in_year is None else self.bus_days_in_year
        return float(calculate_year_fraction(start, end, self.day_count, bus_days, calendar=self.calendar))

    def period(self, start: datetime, end: datetime) -> float:
        """The coupon leg's fraction: today's Phoenix ``calculate_day_count_fraction`` call (plan ruling P-3)."""
        return float(calculate_day_count_fraction(start, end, self.day_count))


def _check_instant(kind: AccrualKind, value, name: str) -> None:
    if value is None:
        return
    if kind is AccrualKind.DATE:
        if not isinstance(value, datetime):
            raise ValidationError(f"{name} of a DATE accrual period must be a datetime, got {value!r}")
        return
    if isinstance(value, (bool, datetime)) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValidationError(f"{name} of a TIME accrual period must be a finite year fraction, got {value!r}")


def _check_fraction(value, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or value < 0:
        raise ValidationError(f"{name} must be a finite non-negative fraction, got {value!r}")


@dataclass(frozen=True)
class KnockInAnchor:
    """The knock-in a REBASED schedule hangs from (spec §5.1): resolved once, when the knock-in happens.

    ``fixing`` is the knock-in record's contractual fixing (its accrual ``end``), not the close it was observed on.
    """

    source_index: int
    fixing: Instant
    origin: Instant
    basis: AccrualBasis

    def __post_init__(self):
        kind = AccrualKind.DATE if isinstance(self.fixing, datetime) else AccrualKind.TIME
        _check_instant(kind, self.fixing, "knock-in fixing")
        _check_instant(kind, self.origin, "knock-in origin")
        if not isinstance(self.basis, AccrualBasis):
            raise ValidationError("a knock-in anchor needs its AccrualBasis")

    def accrual(self) -> float:
        """The contract's accrual from its origin to the knock-in fixing."""
        if self.fixing == self.origin:
            return 0.0
        if isinstance(self.fixing, datetime):
            return self.basis.cumulative(self.origin, self.fixing)
        return float(self.fixing) - float(self.origin)


@dataclass(frozen=True)
class AccrualPeriod:
    """One record's accrual, self-contained (spec §4.1).

    Representations (every other combination raises):
      * ``NONE``: no endpoints; with ``cumulative_fraction`` / ``period_fraction`` overrides, or none at all
        (a non-accruing record).
      * ``TIME`` / ``DATE`` with ``origin`` + ``end``, ``start`` optional (unknown).
      * ``relative`` (``TIME``): ``start`` + ``end`` are quoted offsets from a knock-in anchor, embedded or passed.
    """

    kind: AccrualKind
    origin: Optional[Instant] = None
    start: Optional[Instant] = None
    end: Optional[Instant] = None
    basis: Optional[AccrualBasis] = None
    period_basis: Optional[AccrualBasis] = None
    cumulative_fraction: Optional[float] = None
    period_fraction: Optional[float] = None
    relative: bool = False
    anchor: Optional[KnockInAnchor] = None
    #: The contract terms this period was derived from (``accrual_build``); ``None`` for a period the caller set
    #: explicitly. A rebuilt product keeps a derived period only while its terms are unchanged.
    terms: Optional[tuple] = None

    def __post_init__(self):
        if not isinstance(self.kind, AccrualKind):
            raise ValidationError(f"accrual kind must be AccrualKind, got {self.kind!r}")
        _check_fraction(self.cumulative_fraction, "cumulative_fraction")
        _check_fraction(self.period_fraction, "period_fraction")
        for name in ("origin", "start", "end"):
            _check_instant(self.kind, getattr(self, name), name)
        if self.anchor is not None and not isinstance(self.anchor, KnockInAnchor):
            raise ValidationError("an accrual anchor must be a KnockInAnchor")
        if self.basis is not None and not isinstance(self.basis, AccrualBasis):
            raise ValidationError("an accrual basis must be an AccrualBasis")
        if self.period_basis is None and self.basis is not None:
            object.__setattr__(self, "period_basis", self.basis)
        if self.kind is AccrualKind.NONE:
            if any(v is not None for v in (self.origin, self.start, self.end)) or self.relative or self.anchor:
                raise ValidationError("a NONE accrual period carries no endpoints")
            return
        if self.relative:
            if self.kind is not AccrualKind.TIME or self.origin is not None:
                raise ValidationError("a relative accrual period is TIME offsets with no origin")
            if self.start is None or self.end is None:
                raise ValidationError("a relative accrual period needs start and end offsets")
        else:
            if self.anchor is not None:
                raise ValidationError("only a relative accrual period carries an anchor")
            if self.origin is None or self.end is None:
                raise ValidationError("an accrual period needs its origin and end")
            if self.end < self.origin:
                raise ValidationError(f"accrual end {self.end!r} is before its origin {self.origin!r}")
            if self.start is not None and self.start < self.origin:
                raise ValidationError(f"accrual start {self.start!r} is before its origin {self.origin!r}")
        if self.start is not None and self.end < self.start:
            raise ValidationError(f"accrual end {self.end!r} is before its start {self.start!r}")
        if self.kind is AccrualKind.DATE and self.basis is None:
            raise ValidationError("a DATE accrual period needs its basis")


def cumulative_accrual(period: AccrualPeriod, *, knock_in: Optional[KnockInAnchor] = None) -> float:
    """The cumulative leg (knock-out accrual, terminal tenor) of one period."""
    if period.cumulative_fraction is not None:
        return float(period.cumulative_fraction)
    if period.relative:
        anchor = period.anchor if period.anchor is not None else knock_in
        if anchor is None:
            raise ValidationError("a REBASED accrual period needs its knock-in anchor, embedded or passed")
        return anchor.accrual() + float(period.end)
    if period.kind is AccrualKind.NONE:
        raise ValidationError("this record has no accrual terms (a non-accruing contract)")
    if period.end == period.origin:
        return 0.0
    if period.kind is AccrualKind.DATE:
        return period.basis.cumulative(period.origin, period.end)
    return float(period.end) - float(period.origin)


def coupon_accrual(period: AccrualPeriod) -> float:
    """The coupon leg (Phoenix period fraction) of one period."""
    if period.period_fraction is not None:
        return float(period.period_fraction)
    if period.kind is AccrualKind.NONE:
        raise ValidationError("this record has no accrual terms (a non-accruing contract)")
    if period.start is None:
        raise ValidationError("coupon period start unknown: give the contract a fixed_coupon_year_fraction or "
                              "accrual_factors")
    if period.end == period.start:
        return 0.0
    if period.kind is AccrualKind.DATE:
        return period.period_basis.period(period.start, period.end)
    return float(period.end) - float(period.start)


# -- the resolver: the only place a product's accrual is read (spec §5) ------------------------------------------

def _period_of(record) -> AccrualPeriod:
    period = getattr(record, "accrual", None)
    if period is None:
        raise ValidationError("this observation record carries no accrual period: a schedule set on a built product "
                              "needs product.validate() to derive its periods")
    return period


def _guard_current(product) -> None:
    """Accrual terms changed on a built product without ``validate()`` would pay the old cash: refuse instead."""
    built = getattr(product, "_accrual_terms_key", None)
    if built is None:
        return
    from .accrual_build import accrual_terms_key
    if accrual_terms_key(product) != built:
        raise ValidationError("the contract's accrual terms changed after its periods were derived: call "
                              "product.validate() to derive them again")


def ko_accrual_factor(product, record, env=None, *, knock_in: Optional[KnockInAnchor] = None) -> float:
    """The knock-out leg, in today's precedence: explicit factor, then the annualization flag, then the period."""
    period = _period_of(record)
    if period.cumulative_fraction is not None:
        return float(period.cumulative_fraction)
    if not product._effective_annualized_flag(product.accrual_config.is_annualized_ko):
        return 1.0
    _guard_current(product)
    return cumulative_accrual(period, knock_in=knock_in)


def coupon_period_fraction(product, record, env=None) -> float:
    """The Phoenix coupon leg: 1 for a per-period rate, else the period's fraction."""
    if not product.is_coupon_rate_annualized:
        return 1.0
    period = _period_of(record)
    if period.period_fraction is None:
        _guard_current(product)
    return coupon_accrual(period)


def contract_tenor(product, env=None, leg: str = "contract", *, knock_in: Optional[KnockInAnchor] = None) -> float:
    """The cumulative leg of a terminal period (spec §4.4): ``contract``, or KO-reset's ``pre`` / ``post``."""
    terminals = getattr(product, "_terminal_accrual", None)
    if not terminals or leg not in terminals:
        raise ValidationError(f"{type(product).__name__} carries no {leg!r} terminal accrual period")
    period = terminals[leg]
    if period.cumulative_fraction is None:
        _guard_current(product)
    return cumulative_accrual(period, knock_in=knock_in)


def knock_in_anchor_for(record, source_index: int) -> Optional[KnockInAnchor]:
    """The anchor a knock-in on ``record`` fixes (spec §5.1); ``None`` for a non-accruing record."""
    period = _period_of(record)
    if period.kind is AccrualKind.NONE:
        return None
    return KnockInAnchor(source_index=int(source_index), fixing=period.end, origin=period.origin, basis=period.basis)
