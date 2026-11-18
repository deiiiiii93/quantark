"""
CSI construction from real closes under the five rebalance rules.

Between rebalance dates the index holds fixed share counts nᵢ, so
I(t) = Σᵢ nᵢ·Sᵢ(t).  At the close of a rebalance date the counts reset to
nᵢ = wᵢ·I(t)/Sᵢ(t); the level itself does not move.  Rule dates:

* quarterly — first session of Jan / Apr / Jul / Oct
* monthly   — first session of each month
* weekly    — first session of each ISO week
* daily     — every session (a constant mix)
* none      — never (fixed shares from the base date)

The base date fixes the starting share counts and is not itself a rebalance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

RULES = ("none", "quarterly", "monthly", "weekly", "daily")
TRADING_DAYS = 244          # the engine's trading clock (sessions per year)


@dataclass(frozen=True)
class IndexPath:
    """One rule's index.

    Attributes:
        level: Index level at each session's close.
        shares: Share counts held over the step that starts at each close
            (after any rebalance at that close).
        weights: Current weights nᵢ·Sᵢ/I at each close, same timing as ``shares``.
        rebalance_dates: Sessions whose close reset the counts.
    """

    level: pd.Series
    shares: pd.DataFrame
    weights: pd.DataFrame
    rebalance_dates: pd.DatetimeIndex


def rebalance_dates(calendar: pd.DatetimeIndex, rule: str, start, end) -> pd.DatetimeIndex:
    """Rebalance sessions of ``rule`` strictly after ``start`` and up to ``end`` (inclusive)."""
    if rule not in RULES:
        raise ValidationError(f"unknown rebalance rule {rule!r}; expected one of {RULES}")
    cal = pd.DatetimeIndex(calendar).sort_values()
    window = cal[(cal > pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    if rule == "none" or window.empty:
        return pd.DatetimeIndex([])
    if rule == "daily":
        return window
    # First session of each period, computed on the full calendar so a period that began before
    # ``start`` does not produce a spurious "first" session inside the window.
    if rule == "quarterly":
        key = cal.year * 10 + cal.quarter
    elif rule == "monthly":
        key = cal.year * 100 + cal.month
    else:
        iso = cal.isocalendar()
        key = iso["year"].to_numpy() * 100 + iso["week"].to_numpy()
    firsts = pd.Series(cal, index=cal).groupby(np.asarray(key)).min()
    firsts = pd.DatetimeIndex(firsts.values)
    return firsts[(firsts > pd.Timestamp(start)) & (firsts <= pd.Timestamp(end))].sort_values()


def build_index(prices: pd.DataFrame, weights: pd.Series, rule: str, base_date, base_level: float = 1000.0) -> IndexPath:
    """Build one rule's index from aligned closes (columns in ``weights`` order)."""
    w = weights.astype(float)
    if not is_close(float(w.sum()), 1.0) or (w <= 0.0).any():
        raise ValidationError("target weights must be positive and sum to one")
    px = prices.loc[prices.index >= pd.Timestamp(base_date), list(w.index)].astype(float)
    if px.empty:
        raise ValidationError(f"no prices on or after the base date {base_date}")
    dates = px.index
    reb = rebalance_dates(dates, rule, dates[0], dates[-1])
    is_reb = dates.isin(reb)
    s = px.to_numpy()
    wv = w.to_numpy()
    n = wv * base_level / s[0]
    levels = np.empty(len(dates))
    shares = np.empty_like(s)
    levels[0] = base_level
    shares[0] = n
    for k in range(1, len(dates)):
        level = float(n @ s[k])
        if is_reb[k]:
            n = wv * level / s[k]
        levels[k] = level
        shares[k] = n
    level_s = pd.Series(levels, index=dates, name=rule)
    shares_df = pd.DataFrame(shares, index=dates, columns=w.index)
    weights_df = shares_df * px / level_s.to_numpy()[:, None]
    return IndexPath(level_s, shares_df, weights_df, pd.DatetimeIndex(dates[is_reb]))

