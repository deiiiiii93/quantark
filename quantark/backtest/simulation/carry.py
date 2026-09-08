"""Constant-maturity carry curves and the listed IM chain (spec 4.1, 6).

``B(T) = ln(F(T)/S)`` on a fixed tenor grid, piecewise-linear in ``T``
with ``B(0) = 0`` and the last segment's slope continued beyond the last
node -- the ``ForwardCarryCurve`` rules, vectorised over paths and days.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from quantark.param.div.forward_carry_curve import ForwardCarryCurve
from quantark.util.exceptions import ValidationError

from .paths.market_path import MarketPath


def carry_at(carry: np.ndarray, tenor_grid: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """Read ``B`` at ``tenors`` from curves given on ``tenor_grid``.

    ``carry`` has shape ``(..., n_tenors)``; the result has shape
    ``(..., len(tenors))``.  Interpolation is linear in ``B`` against ``T``
    (constant forward carry between nodes); beyond the last node the last
    segment continues; ``B(0) = 0``.
    """
    grid = np.concatenate([[0.0], np.asarray(tenor_grid, dtype=float)])
    b = np.asarray(carry, dtype=float)
    b = np.concatenate([np.zeros(b.shape[:-1] + (1,)), b], axis=-1)
    t = np.asarray(tenors, dtype=float)
    if np.any(t < 0.0):
        raise ValidationError("tenors must be non-negative")
    idx = np.searchsorted(grid, t, side="right") - 1
    idx = np.clip(idx, 0, grid.size - 2)
    t0, t1 = grid[idx], grid[idx + 1]
    w = (t - t0) / (t1 - t0)                       # w > 1 beyond the last node: slope continues
    b0, b1 = b[..., idx], b[..., idx + 1]
    return b0 + w * (b1 - b0)


def curve_from_chain(
    spot: float, tenors: Sequence[float], prices: Sequence[float], tenor_grid: np.ndarray
) -> np.ndarray:
    """The constant-maturity curve implied by a listed chain (history side).

    Contracts with a non-positive tenor (expiring today or earlier) are
    dropped; the remaining ``(T_i, ln(F_i/spot))`` nodes form a
    ``ForwardCarryCurve`` sampled at ``tenor_grid``.
    """
    if spot <= 0.0:
        raise ValidationError("spot must be positive")
    nodes = [
        (float(t), float(np.log(float(p) / spot)))
        for t, p in zip(tenors, prices)
        if float(t) > 0.0
    ]
    if not nodes:
        raise ValidationError("curve_from_chain needs at least one contract with a positive tenor")
    nodes.sort()
    curve = ForwardCarryCurve(nodes)
    return np.array([curve.carry(float(t)) for t in np.asarray(tenor_grid, dtype=float)])


_QUARTERLY = (3, 6, 9, 12)


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7      # Friday is weekday 4
    return first + timedelta(days=offset + 14)


def im_expiry(year: int, month: int, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """Third Friday of the month, or the next trading day in ``calendar`` if it does not trade."""
    friday = pd.Timestamp(third_friday(year, month))
    if friday in calendar:
        return friday
    later = calendar[calendar > friday]
    if len(later) == 0:
        raise ValidationError(f"calendar ends before the IM{year % 100:02d}{month:02d} expiry {friday.date()}")
    return pd.Timestamp(later[0])


def _next_month(year: int, month: int) -> Tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def listed_im_contracts(day: pd.Timestamp, calendar: pd.DatetimeIndex) -> List[Tuple[str, pd.Timestamp]]:
    """The four IM contracts CFFEX lists on ``day``: current month, next month, next two quarterlies."""
    day = pd.Timestamp(day).normalize()
    year, month = day.year, day.month
    if im_expiry(year, month, calendar) < day:       # this month's contract has expired
        year, month = _next_month(year, month)
    months = [(year, month)]
    months.append(_next_month(year, month))
    y, m = months[-1]
    while len(months) < 4:
        y, m = _next_month(y, m)
        if m in _QUARTERLY:
            months.append((y, m))
    return [(f"IM{y % 100:02d}{m:02d}", im_expiry(y, m, calendar)) for y, m in months]
