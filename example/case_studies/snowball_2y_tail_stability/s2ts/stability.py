"""q(2Y) day-to-day stability (design §4.3): series, day types, switch floor, summaries.

``q_d(2Y) = r − B_d(2.0)/2.0`` at a constant 2.0Y tenor, the curve rebuilt each
day from that day's chain.  Changes are reported in bp of yield.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from quantark.param.curve.carry_curve import CarryCurve
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_exp, safe_log

from . import RATE, StudyDataError
from .chain import DayChain
from .tails import build_curve

Q2Y_TENOR = 2.0
BP = 1e4
DAY_TYPES = ("ordinary", "mo_roll", "atm_switch")


def q2y(curve: CarryCurve) -> float:
    """``r − B(2.0)/2.0``."""
    return RATE - float(curve.carry(Q2Y_TENOR)) / Q2Y_TENOR


def day_types(chains: Sequence[DayChain]) -> Dict[date, str]:
    """Type of the change INTO each day (the first day has none).

    ``mo_roll``: the longest MO series changed; ``atm_switch``: same series, a new
    ATM strike; ``ordinary`` otherwise.  A roll wins when both happen.
    """
    out: Dict[date, str] = {}
    for prev, dc in zip(chains, chains[1:]):
        if dc.mo.series != prev.mo.series:
            out[dc.day] = "mo_roll"
        elif dc.mo.strike != prev.mo.strike:
            out[dc.day] = "atm_switch"
        else:
            out[dc.day] = "ordinary"
    return out


def _df(dc: DayChain) -> float:
    return float(safe_exp(-RATE * dc.mo.tenor))


def _parity_forward(dc: DayChain, strike: float) -> float:
    if strike not in dc.mo_all:
        raise StudyDataError(f"{dc.day}: {dc.mo.series} has no settled pair at K={strike:g}")
    c, p = dc.mo_all[strike]
    return strike + (c - p) / _df(dc)


def multi_strike_forward(dc: DayChain, strikes: Optional[Iterable[float]] = None) -> float:
    """Mean of ``K + (C−P)/DF`` over ``strikes`` (default: the 5 strikes nearest ATM).

    Equal-weight least squares of ``C − P = DF·(F − K)`` with ``DF`` known.
    """
    ks = [k for k, _, _ in dc.mo.strikes_near] if strikes is None else list(strikes)
    if not ks:
        raise ValidationError("multi_strike_forward needs at least one strike")
    return float(np.mean([_parity_forward(dc, k) for k in ks]))


def mo_forward(dc: DayChain, multi_strike: bool = False) -> float:
    """The MO pillar forward under the base (ATM pair) or the multi-strike rule."""
    return multi_strike_forward(dc) if multi_strike else dc.mo.forward


def settlement_chain(dc: DayChain) -> DayChain:
    """The all-settlement variant: IM pillars (and the far-IM carry) at CFFEX settlement.

    The MO pillar is already at settlement and spot stays the index close.
    Each ``ImQuote.close`` is replaced by its settlement so every consumer of the
    chain (curve, anchor) reads the settlement mark.
    """
    im = tuple(dataclasses.replace(q, close=q.settle) for q in dc.im)
    far = im[-1]
    return dataclasses.replace(dc, im=im, far_carry=float(safe_log(far.close / dc.spot)) / far.tenor)


def switch_floor(prev: DayChain, dc: DayChain, tail: str, b_inf: Optional[float], *,
                 multi_strike: bool = False) -> float:
    """q(2Y) noise of an ATM switch alone (same day's settlements).

    q2y with today's MO pillar priced at yesterday's ATM strike (or, with
    ``multi_strike``, yesterday's 5-strike set) minus q2y at today's.  Zero when
    the strike did not change; undefined across an MO roll.
    """
    if prev.mo.series != dc.mo.series:
        raise ValidationError(f"{dc.day}: switch floor undefined across an MO roll "
                              f"({prev.mo.series} -> {dc.mo.series})")
    if multi_strike:
        f_prev = multi_strike_forward(dc, [k for k, _, _ in prev.mo.strikes_near])
        f_now = multi_strike_forward(dc)
    else:
        f_prev = _parity_forward(dc, prev.mo.strike)
        f_now = dc.mo.forward
    return q2y(build_curve(dc, tail, b_inf, mo_forward=f_prev)) - q2y(build_curve(dc, tail, b_inf, mo_forward=f_now))


def q2y_series(chains: Sequence[DayChain], tail: str, anchors: Mapping[date, float], *,
               days: Optional[Iterable[date]] = None, multi_strike: bool = False) -> Dict[date, float]:
    """``{d: q2y}`` over ``days`` (default: every day with an anchor)."""
    wanted = set(anchors) if days is None else set(days)
    return {
        dc.day: q2y(build_curve(dc, tail, anchors.get(dc.day), mo_forward=mo_forward(dc, multi_strike)))
        for dc in chains if dc.day in wanted
    }


def _stats(x: List[float]) -> Dict[str, Optional[float]]:
    a = np.asarray(x, dtype=float)
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "mad": None, "p99_abs": None, "sumsq": 0.0}
    return {
        "n": n,
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if n >= 2 else None,
        # median absolute deviation from the median
        "mad": float(np.median(np.abs(a - np.median(a)))),
        "p99_abs": float(np.percentile(np.abs(a), 99)),
        "sumsq": float(np.sum(a * a)),
    }


def summarize(series: Mapping[date, float], types: Mapping[date, str]) -> Dict[str, Dict]:
    """``std``, ``mad`` (median absolute deviation), ``p99_abs`` of daily Δq in bp of yield.

    Overall (``all``) and per day type; ``sumsq_share`` is each type's share of
    ``Σ Δ²`` (how much of the movement the mechanical days carry).  Δ is taken
    between consecutive dates of ``series``.
    """
    days = sorted(series)
    by_type: Dict[str, List[float]] = {t: [] for t in DAY_TYPES}
    every: List[float] = []
    for a, b in zip(days, days[1:]):
        dq = BP * (series[b] - series[a])
        kind = types.get(b)
        if kind not in by_type:
            raise ValidationError(f"no day type for {b}")
        by_type[kind].append(dq)
        every.append(dq)
    out: Dict[str, Dict] = {"all": _stats(every)}
    total = out["all"]["sumsq"]
    for kind, vals in by_type.items():
        s = _stats(vals)
        s["sumsq_share"] = s["sumsq"] / total if total > 0.0 else None
        out[kind] = s
    return out


def floor_distribution(floors: Sequence[float]) -> Dict[str, Optional[float]]:
    """Switch-floor distribution in bp of yield: median and p90 of ``|floor|``, signed std."""
    a = BP * np.asarray(floors, dtype=float)
    if a.size == 0:
        return {"n": 0, "median_abs": None, "p90_abs": None, "std": None}
    return {
        "n": int(a.size),
        "median_abs": float(np.median(np.abs(a))),
        "p90_abs": float(np.percentile(np.abs(a), 90)),
        "std": float(a.std(ddof=1)) if a.size >= 2 else None,
    }
