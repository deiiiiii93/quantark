"""Clock-wrapped market objects in the factor model (patch spec 2026-09-03 §8).

A TradingClock wrapper (``TradingClockVolSurface``, ``TradingClockRateCurve``,
``TradingClockDividendYield``) re-expresses an inner object through a
``BusinessTimeMap`` anchored at the environment's valuation date. Two wrappers
are the same MARKET when their class, inner object and clock agree; the anchor
is the valuation date's business and moves with TIME, so every scenario state
re-anchors its wrappers at its own valuation date (sticky trading tenor).
"""
from __future__ import annotations

from typing import Any, Callable

from quantark.pnlexplain.equity.fingerprints import calendars_equal
from quantark.util.exceptions import ValidationError

CLOCK_FIELDS = ("vol_surface", "rate_curve", "div_yield")


def is_clock_wrapped(obj: Any) -> bool:
    """True for a TradingClock wrapper (has ``inner`` and a ``time_map``)."""
    return obj is not None and getattr(obj, "time_map", None) is not None and hasattr(obj, "inner")


def clocks_equal(a: Any, b: Any) -> bool:
    """Same days_per_year and a semantically equal calendar."""
    return int(a.days_per_year) == int(b.days_per_year) and calendars_equal(a.calendar, b.calendar)


def wrapped_equal(a: Any, b: Any, inner_equal: Callable[[Any, Any], bool]) -> bool:
    """Same wrapper class, same clock, equal inner objects; the map's anchor is NOT compared."""
    if type(a) is not type(b):
        return False
    return clocks_equal(a.time_map.clock, b.time_map.clock) and inner_equal(a.inner, b.inner)


def validate_clock_env(env: Any, label: str) -> None:
    """Every wrapped field's map must be anchored at the environment's valuation date."""
    for name in CLOCK_FIELDS:
        obj = getattr(env, name, None)
        if is_clock_wrapped(obj) and obj.time_map.anchor_date != env.valuation_date:
            raise ValidationError(
                f"{label}.{name} is clock-wrapped with a time map anchored at {obj.time_map.anchor_date}, "
                f"not at the environment's valuation date {env.valuation_date}"
            )


def validate_clock_pair(e0: Any, e1: Any) -> None:
    """The same fields are wrapped on both sides and their clocks agree: a clock change is not a market move."""
    for name in CLOCK_FIELDS:
        a, b = getattr(e0, name, None), getattr(e1, name, None)
        wa, wb = is_clock_wrapped(a), is_clock_wrapped(b)
        if wa != wb:
            raise ValidationError(f"{name} is clock-wrapped on one side only: a clock change is not a market move")
        if wa and not clocks_equal(a.time_map.clock, b.time_map.clock):
            raise ValidationError(f"{name} carries two different trading clocks: a clock change is not a market move")


def re_anchor(env: Any) -> None:
    """Re-anchor every wrapped field at ``env.valuation_date`` (mutates the given private copy)."""
    for name in CLOCK_FIELDS:
        obj = getattr(env, name, None)
        if is_clock_wrapped(obj) and obj.time_map.anchor_date != env.valuation_date:
            setattr(env, name, obj.with_time_map(obj.time_map.re_anchored(env.valuation_date)))
