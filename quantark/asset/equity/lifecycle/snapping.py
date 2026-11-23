"""Snapped numeric observation times (spec 2026-10-08 KO-reset lifecycle, section 4a).

The tracker observes a numeric record on the close ``round(time * 365)`` days after its start; a
pricing copy, a surface or an engine that kept the fractional time saw the record a fraction of a day
away -- still ahead on the close that had observed it. Snapping the contract's numeric times to that
day makes every consumer agree with the tracker. A time already on a whole day keeps its exact float,
so whole-day contracts are bit-identical.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import List, Optional

from quantark.asset.equity.product.option.contract_clock import canonicalize_contract_clock
from quantark.util.numerical import Tolerance

DAYS_PER_YEAR = 365.0

__all__ = ["is_off_day", "snap_contract", "snap_time"]


def is_off_day(time: float) -> bool:
    """Whether ``time`` (years) lies off a whole day by more than ``Tolerance.PRECISION`` days."""
    days = float(time) * DAYS_PER_YEAR
    return abs(days - round(days)) > Tolerance.PRECISION


def snap_time(time: Optional[float]) -> Optional[float]:
    """``round(time * 365) / 365`` (the rounding ``_record_is_due`` uses); a whole-day time unchanged."""
    if time is None or not is_off_day(time):
        return time
    return round(float(time) * DAYS_PER_YEAR) / DAYS_PER_YEAR


def _snapped_times(times: List[Optional[float]]) -> List[Optional[float]]:
    """One schedule's numeric times snapped, kept in order.

    An off-day time moves to its rounded day. A whole-day time keeps its exact float, which may sit an ulp
    either side of ``k / 365``; an off-day neighbour rounding onto the same day then takes that kept float
    (the lowest kept one when it lies before the day, the highest when after), so the schedule stays
    non-decreasing and whole-day times stay bit-identical.
    """
    kept: dict = {}
    for t in times:
        if t is not None and not is_off_day(t):
            kept.setdefault(int(round(float(t) * DAYS_PER_YEAR)), []).append(t)
    out: List[Optional[float]] = []
    for t in times:
        if t is None or not is_off_day(t):
            out.append(t)
            continue
        days = float(t) * DAYS_PER_YEAR
        day = int(round(days))
        same_day = kept.get(day)
        if same_day:
            out.append(min(same_day) if days < day else max(same_day))
        else:
            out.append(day / DAYS_PER_YEAR)
    return out


def _snap_record(record, snapped: Optional[float]):
    """``record`` at its snapped numeric time (dated and whole-day records unchanged)."""
    time = record.observation_time
    if record.observation_date is not None or time is None or not is_off_day(time):
        return record
    settlement = record.settlement_time
    if settlement is not None:
        settlement = snapped + (float(settlement) - float(time))   # the lag is kept
    return replace(record, observation_time=snapped, settlement_time=settlement)


def _snap_schedule(schedule):
    if schedule is None:
        return None
    numeric = [None if r.observation_date is not None else r.observation_time for r in schedule.records]
    snapped = _snapped_times(numeric)
    records = [_snap_record(r, t) for r, t in zip(schedule.records, snapped)]
    if all(a is b for a, b in zip(records, schedule.records)):
        return schedule
    return replace(schedule, records=records)


def _snap_dates(dates: Optional[List[float]]):
    if dates is None or not any(isinstance(t, (int, float)) and is_off_day(t) for t in dates):
        return dates
    numeric = [t if isinstance(t, (int, float)) else None for t in dates]
    snapped = _snapped_times(numeric)
    return [s if isinstance(t, (int, float)) else t for t, s in zip(dates, snapped)]


def _snap_config(config):
    if config is None:
        return None
    changes = {
        "ko_observation_schedule": _snap_schedule(config.ko_observation_schedule),
        "ki_observation_schedule": _snap_schedule(config.ki_observation_schedule),
        "ko_observation_dates": _snap_dates(config.ko_observation_dates),
        "ki_observation_dates": _snap_dates(config.ki_observation_dates),
    }
    if all(changes[name] is getattr(config, name) for name in changes):
        return config
    return replace(config, **changes)


def _keep_schedule_two_last(contract) -> None:
    """A numeric REBASED KO-reset keeps "schedule 2 ends last" (spec D35) on the day grid.

    Rounding is monotone, so snapped ABSOLUTE dates keep their order; a REBASED end is a SUM, the first knock-in plus
    the last offset, and two rounded terms can fall one day short of the rounded schedule-1 end on a contract booked
    exactly on the boundary with off-day times (91.4 + 273.4 = 364.8 days rounds to 91 + 273 < 365). The rule holds on
    the booked times; on the day grid the last offset then moves to the next day, the one day ``round`` took off, so the
    snapped contract -- and every knocked-in copy built from it -- stays valid. Never more than one day; a contract whose
    snapped end already reaches schedule 1's is unchanged.
    """
    from quantark.util.enum import PostKOScheduleMode
    from quantark.util.numerical import is_close

    if getattr(contract, "post_ko_mode", None) != PostKOScheduleMode.REBASED:
        return
    pre = contract.barrier_config.ko_observation_schedule
    ki = contract.barrier_config.ki_observation_schedule
    post = contract.post_barrier_config.ko_observation_schedule
    if any(s is None or not s.records for s in (pre, ki, post)):
        return
    times = [r.observation_time for s in (pre, ki, post) for r in s.records]
    if any(t is None for t in times):
        return
    first = min(float(r.observation_time) for r in ki.records)
    last_index = max(range(len(post.records)), key=lambda i: float(post.records[i].observation_time))
    last = float(post.records[last_index].observation_time)
    pre_end = max(float(r.observation_time) for r in pre.records)
    if first + last >= pre_end or is_close(first + last, pre_end, abs_tol=Tolerance.ZERO):
        return
    moved_to = (round(last * DAYS_PER_YEAR) + 1) / DAYS_PER_YEAR
    record = post.records[last_index]
    settlement = record.settlement_time
    if settlement is not None:
        settlement = moved_to + (float(settlement) - last)              # the lag is kept
    records = list(post.records)
    records[last_index] = replace(record, observation_time=moved_to, settlement_time=settlement)
    contract.post_barrier_config = replace(contract.post_barrier_config,
                                           ko_observation_schedule=replace(post, records=records))


def snap_contract(product):
    """A deep copy of ``product`` with every numeric observation time and a numeric maturity snapped."""
    contract = deepcopy(product)
    moved = False
    # a dated REBASED KO-reset's schedule-2 offsets are whole calendar months, a product term, not times to snap
    dated_rebased = getattr(contract, "_dated_rebased", lambda: False)()
    for name in ("barrier_config", "post_barrier_config"):
        if dated_rebased and name == "post_barrier_config":
            continue
        config = getattr(contract, name, None)
        if config is not None:
            snapped = _snap_config(config)
            moved = moved or snapped is not config
            setattr(contract, name, snapped)
    if moved and not dated_rebased:
        _keep_schedule_two_last(contract)
    maturity = getattr(contract, "maturity", None)
    if getattr(contract, "exercise_date", None) is None and isinstance(maturity, (int, float)) and is_off_day(maturity):
        contract.maturity = snap_time(float(maturity))
        moved = True
    if moved and getattr(contract, "_terminal_accrual", None):
        # Snapping moves observation timing only: the terminal tenors (V0, V1) stay the contract's. A numeric terminal
        # is derived from the last fixing or the maturity, so a later validate() (QUAD V2 validates every product)
        # would otherwise re-derive it from the snapped times (0.5 -> 182/365). Records already keep their periods.
        contract._terminal_accrual_explicit = True
    # The day-snap rounds PER schedule: a kept whole-day float one ulp off k/365 in one schedule and a neighbour
    # rounded onto k/365 in another are one instant in two spellings. The contract clock takes them to one float,
    # last, after the maturity snap so the snapped maturity is the pin.
    if canonicalize_contract_clock(contract) and getattr(contract, "_terminal_accrual", None):
        contract._terminal_accrual_explicit = True
    return contract
