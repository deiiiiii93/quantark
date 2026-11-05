"""Floating-rate projection: the simple forward on a curve over the index window.

Spec: ``.ship/2026-11-05-curve-construction/spec.md`` §Decisions "Projection
convention" and "Missing fixings and running periods".

A floating coupon pays ``N · L · dcf`` with ``L`` a *simple* rate, so the rate
projected from a curve must be the simple forward over the window the index
covers, ``L = (P(t_a)/P(t_b) − 1) / dcf_w`` on the projection curve ``P``. The
rate engines (IRS / FRN legs, FRA, cap/floor, the swaption fallback) and the
curve bootstrapper all call :func:`simple_forward_rate`, so a curve bootstrapped
from swap quotes reprices those swaps through the engines by construction.
The former code plugged the continuously-compounded forward over
``[t_fix, t_fix + tenor]`` (overnight: the zero rate at the fixing date) into
the simple formula. The move depends on the index: compounding convexity only
(``≈ f²τ/2``, about 1 bp at 3 % / 3M) for a term index on the axis day count;
the day-count basis term (``≈ −f·5/365``) dominates on ACT/360 indices
(−3.8 bp at 5 % / 3M); overnight-compounded legs move first-order in the
curve slope (≈ 200 bp on a 10Y quarterly coupon with ``z(t) = 2 % + 0.2 %·t``).

Time axis: calendar ACT/365F from the valuation date, ``(date − valuation).days / 365``,
the convention every rate engine uses.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

from quantark.param.rrf.rate_curve import RateCurve
from quantark.util.exceptions import ValidationError


def act365f(valuation_date: datetime, date: datetime) -> float:
    """Year fraction on the curve axis: ``(date − valuation_date).days / 365``."""
    return (date - valuation_date).days / 365.0


def simple_forward_rate(
    curve: RateCurve,
    t_start: float,
    t_end: float,
    dcf: Optional[float] = None,
) -> float:
    """Simple forward rate on ``curve`` over ``[t_start, t_end]`` in the basis ``dcf``.

    Args:
        curve: projection curve (any ``RateCurve``).
        t_start: window start in years; may be negative for a running period.
        t_end: window end in years, ``> t_start`` and ``> 0``.
        dcf: the coupon's day-count fraction for the window; ``None`` means the
            window's own axis length ``t_end − t_start``.

    Returns:
        ``t_start >= 0``: ``(DF(t_start)/DF(t_end) − 1) / dcf``.
        ``t_start < 0`` (period already running, fixing not in the store): the
        spot simple rate from the valuation date to ``t_end`` in the coupon's
        basis, ``(1/DF(t_end) − 1) / (dcf · t_end/(t_end − t_start))`` — the
        coupon's ``dcf`` pro-rated to the remaining window.
    """
    t_start = float(t_start)
    t_end = float(t_end)
    if not (math.isfinite(t_start) and math.isfinite(t_end)):
        raise ValidationError("window times must be finite")
    if t_end <= t_start:
        raise ValidationError(f"t_end ({t_end}) must be greater than t_start ({t_start})")
    if t_end <= 0.0:
        raise ValidationError(f"t_end ({t_end}) must be positive: the window is in the past")
    window = t_end - t_start
    basis = window if dcf is None else float(dcf)
    if not math.isfinite(basis) or basis <= 0.0:
        raise ValidationError(f"dcf must be positive and finite, got {dcf}")
    df_end = float(curve.get_discount_factor(t_end))
    if t_start >= 0.0:
        df_start = float(curve.get_discount_factor(t_start))
        return (df_start / df_end - 1.0) / basis
    basis_remaining = basis * t_end / window
    return (1.0 / df_end - 1.0) / basis_remaining


def project_floating_rate(
    curve: RateCurve,
    valuation_date: datetime,
    accrual_start: datetime,
    accrual_end: datetime,
    fixing_date: datetime,
    day_count_fraction: float,
    index_tenor_years: float,
    is_overnight: bool,
    in_arrears: bool,
) -> float:
    """Projected simple rate for one floating coupon, per the window rule.

    In-advance resets and every overnight (compounded) index use the accrual
    window ``[accrual_start, accrual_end]`` with the coupon's own day-count
    fraction. A term index reset in arrears covers its own period
    ``[fixing, fixing + tenor]`` (axis-length basis, no convexity adjustment).
    A window that is already running is handled by :func:`simple_forward_rate`'s
    ``t_start < 0`` branch. Callers decide what to do with coupons whose accrual
    has fully elapsed (``MarketDataError`` when still payable) before calling.
    """
    if is_overnight or not in_arrears:
        return simple_forward_rate(
            curve,
            act365f(valuation_date, accrual_start),
            act365f(valuation_date, accrual_end),
            day_count_fraction,
        )
    t_fix = act365f(valuation_date, fixing_date)
    return simple_forward_rate(curve, t_fix, t_fix + float(index_tenor_years))


def missing_fixing_message(
    index_name: str, fixing_date: datetime, accrual_start: datetime, accrual_end: datetime
) -> str:
    """Error text for a payable coupon whose accrual has elapsed without a stored fixing."""
    return (
        f"missing fixing for {index_name} fixed {fixing_date:%Y-%m-%d} "
        f"(accrual {accrual_start:%Y-%m-%d}..{accrual_end:%Y-%m-%d}): the coupon is "
        "determined by a historical fixing the fixing store does not have"
    )
