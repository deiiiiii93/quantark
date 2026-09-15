"""Calendar-time primitives for intraday mode.

The numerical axis is ACT/365 calendar time measured in SECONDS from the
valuation timestamp: tau = seconds / (365 * 86400). This is the only place
that constant lives. Day-level code (``calculate_year_fraction``) truncates to
whole days and is never used on the intraday path.

All arithmetic and ordering goes through UTC: Python subtracts, adds and
compares two datetimes that share one ``tzinfo`` object on the naive wall
clock, which is off by the DST shift in a zone such as America/New_York.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from quantark.util.exceptions import ValidationError

SECONDS_PER_YEAR: float = 365.0 * 86400.0


def require_aware(value: datetime, name: str) -> datetime:
    """Return ``value`` unchanged if it is a timezone-aware datetime, else raise."""
    if not isinstance(value, datetime):
        raise ValidationError(f"{name} must be a timezone-aware datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{name} must be timezone-aware (naive datetime {value!r} rejected)")
    return value


def to_utc(value: datetime, name: str = "timestamp") -> datetime:
    """The same instant expressed in UTC: the key for instant-based arithmetic and ordering."""
    return require_aware(value, name).astimezone(timezone.utc)


def seconds_between(start: datetime, end: datetime) -> float:
    """Signed elapsed seconds from ``start`` to ``end`` (instant-based, DST-safe)."""
    return (to_utc(end, "end") - to_utc(start, "start")).total_seconds()


def calendar_year_fraction(start: datetime, end: datetime) -> float:
    """Signed seconds-exact ACT/365 fraction from ``start`` to ``end`` (instant-based)."""
    return seconds_between(start, end) / SECONDS_PER_YEAR


def add_year_fraction(start: datetime, tau: float) -> datetime:
    """The instant ``tau`` ACT/365 calendar years after ``start``, in ``start``'s zone."""
    moved = to_utc(start, "start") + timedelta(seconds=float(tau) * SECONDS_PER_YEAR)
    return moved.astimezone(start.tzinfo)


def same_instant(a: datetime, b: datetime) -> bool:
    """Whether two aware datetimes denote the same instant (zone-independent)."""
    return to_utc(a, "a") == to_utc(b, "b")
