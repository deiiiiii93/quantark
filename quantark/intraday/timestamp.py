"""Calendar-time primitives for intraday mode.

The numerical axis is ACT/365 calendar time measured in SECONDS from the
valuation timestamp: tau = seconds / (365 * 86400). This is the only place
that constant lives. Day-level code (``calculate_year_fraction``) truncates to
whole days and is never used on the intraday path.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from quantark.util.exceptions import ValidationError

SECONDS_PER_YEAR: float = 365.0 * 86400.0


def require_aware(value: datetime, name: str) -> datetime:
    """Return ``value`` unchanged if it is a timezone-aware datetime, else raise."""
    if not isinstance(value, datetime):
        raise ValidationError(f"{name} must be a timezone-aware datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{name} must be timezone-aware (naive datetime {value!r} rejected)")
    return value


def calendar_year_fraction(start: datetime, end: datetime) -> float:
    """Signed seconds-exact ACT/365 fraction from ``start`` to ``end`` (instant-based)."""
    require_aware(start, "start")
    require_aware(end, "end")
    return (end - start).total_seconds() / SECONDS_PER_YEAR


def add_year_fraction(start: datetime, tau: float) -> datetime:
    """The instant ``tau`` ACT/365 calendar years after ``start``."""
    require_aware(start, "start")
    return start + timedelta(seconds=float(tau) * SECONDS_PER_YEAR)


def same_instant(a: datetime, b: datetime) -> bool:
    """Whether two aware datetimes denote the same instant (zone-independent)."""
    return require_aware(a, "a") == require_aware(b, "b")
