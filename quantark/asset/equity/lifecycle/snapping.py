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


def _snap_record(record):
    time = record.observation_time
    if record.observation_date is not None or time is None or not is_off_day(time):
        return record
    snapped = round(float(time) * DAYS_PER_YEAR) / DAYS_PER_YEAR
    settlement = record.settlement_time
    if settlement is not None:
        settlement = snapped + (float(settlement) - float(time))   # the lag is kept
    return replace(record, observation_time=snapped, settlement_time=settlement)


def _snap_schedule(schedule):
    if schedule is None:
        return None
    records = [_snap_record(r) for r in schedule.records]
    if all(a is b for a, b in zip(records, schedule.records)):
        return schedule
    return replace(schedule, records=records)


def _snap_dates(dates: Optional[List[float]]):
    if dates is None or not any(isinstance(t, (int, float)) and is_off_day(t) for t in dates):
        return dates
    return [snap_time(t) if isinstance(t, (int, float)) else t for t in dates]


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


def snap_contract(product):
    """A deep copy of ``product`` with every numeric observation time and a numeric maturity snapped."""
    contract = deepcopy(product)
    if getattr(contract, "barrier_config", None) is not None:
        contract.barrier_config = _snap_config(contract.barrier_config)
    if getattr(contract, "post_barrier_config", None) is not None:
        contract.post_barrier_config = _snap_config(contract.post_barrier_config)
    maturity = getattr(contract, "maturity", None)
    if getattr(contract, "exercise_date", None) is None and isinstance(maturity, (int, float)):
        contract.maturity = snap_time(float(maturity)) if is_off_day(maturity) else maturity
    return contract
