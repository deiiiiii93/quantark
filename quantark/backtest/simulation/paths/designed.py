"""Designed paths from the dynamic-scenario ``PathBuilder`` (spec 5.3)."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.stresstest.stress.stress_types import StressType
from quantark.util.exceptions import ValidationError

from .market_path import MarketPath, StartState, _validate_tenor_grid

_PARAMS = ("spot", "volatility", "rate", "basis")


def market_path_from_day_path(
    day_path: DayPath, *, start: StartState, calendar: pd.DatetimeIndex, tenor_grid: np.ndarray
) -> MarketPath:
    """One path: day ``d`` is the state after ``DayStep d``'s changes (day 0 applies to ``start``)."""
    grid = _validate_tenor_grid(tenor_grid)
    if start.carry.size != grid.size:
        raise ValidationError("start.carry must be on tenor_grid")
    steps = sorted(day_path.steps, key=lambda s: s.day_index)
    n_days = len(steps)
    if n_days < 1:
        raise ValidationError("a designed path needs at least one DayStep")
    if n_days > len(calendar):
        raise ValidationError(f"calendar has {len(calendar)} days, the path has {n_days}")
    spot, vol, rate = float(start.spot), float(start.atm_vol), float(start.rate)
    carry = start.carry.astype(float).copy()
    out_spot, out_vol, out_rate, out_carry = [], [], [], []
    for step in steps:
        for change in step.changes:
            if change.parameter not in _PARAMS:
                raise ValidationError(f"designed paths support {_PARAMS}, got {change.parameter!r}")
            if change.parameter == "spot":
                spot = float(change.apply(spot))
            elif change.parameter == "volatility":
                vol = float(change.apply(vol))
            elif change.parameter == "rate":
                rate = float(change.apply(rate))
            else:  # basis: a change in the annualised carry
                if change.stress_type == StressType.ABSOLUTE:
                    carry = carry + float(change.stress_value) * grid
                elif change.stress_type == StressType.VALUE:
                    carry = float(change.stress_value) * grid
                else:
                    raise ValidationError("basis changes must be ABSOLUTE or VALUE (annualised carry)")
        out_spot.append(spot)
        out_vol.append(vol)
        out_rate.append(rate)
        out_carry.append(carry.copy())
    meta = {"generator": "designed", "scenario_name": day_path.name, "description": day_path.description,
            "start": {"spot": start.spot, "atm_vol": start.atm_vol, "rate": start.rate}}
    return MarketPath(dates=pd.DatetimeIndex(calendar)[:n_days], spot=np.array([out_spot]),
                      atm_vol=np.array([out_vol]), rate=np.array([out_rate]), carry=np.array([out_carry]),
                      tenor_grid=grid.copy(), meta=meta)


def _hold(n: int, start_index: int) -> List[DayStep]:
    return [DayStep(i, []) for i in range(start_index, n)]


def _check(move_days: int, total_days: int) -> None:
    if move_days < 1 or total_days < move_days:
        raise ValidationError("total_days must be at least the number of moving days, both positive")


class SnowballStressLibrary:
    """Named adverse paths for a snowball; each returns a ``DayPath`` of ``total_days`` steps."""

    @staticmethod
    def crash_into_ki(depth: float, days: int, total_days: int) -> DayPath:
        _check(days, total_days)
        daily = (1.0 - depth) ** (1.0 / days) - 1.0
        steps = [DayStep(i, [ParameterChange("spot", StressType.PERCENTAGE, daily)]) for i in range(days)]
        return DayPath(name=f"crash_into_ki_{depth:.0%}_{days}d", steps=steps + _hold(total_days, days),
                       description=f"spot falls {depth:.0%} over {days} days, then holds")

    @staticmethod
    def v_shape(depth: float, days_down: int, days_up: int, total_days: int) -> DayPath:
        _check(days_down + days_up, total_days)
        down = (1.0 - depth) ** (1.0 / days_down) - 1.0
        up = (1.0 / (1.0 - depth)) ** (1.0 / days_up) - 1.0
        steps = [DayStep(i, [ParameterChange("spot", StressType.PERCENTAGE, down)]) for i in range(days_down)]
        steps += [DayStep(days_down + i, [ParameterChange("spot", StressType.PERCENTAGE, up)]) for i in range(days_up)]
        return DayPath(name=f"v_shape_{depth:.0%}_{days_down}d_{days_up}d",
                       steps=steps + _hold(total_days, days_down + days_up),
                       description="spot falls then fully recovers")

    @staticmethod
    def vol_spike(vol_up: float, decay_days: int, total_days: int) -> DayPath:
        _check(decay_days + 1, total_days)
        steps = [DayStep(0, [ParameterChange("volatility", StressType.ABSOLUTE, vol_up)])]
        steps += [DayStep(i, [ParameterChange("volatility", StressType.ABSOLUTE, -vol_up / decay_days)])
                  for i in range(1, decay_days + 1)]
        return DayPath(name=f"vol_spike_{vol_up:.0%}_{decay_days}d", steps=steps + _hold(total_days, decay_days + 1),
                       description="vol jumps then decays linearly back")

    @staticmethod
    def basis_blowout(carry_shift: float, days: int, total_days: int) -> DayPath:
        _check(days, total_days)
        steps = [DayStep(i, [ParameterChange("basis", StressType.ABSOLUTE, carry_shift / days)]) for i in range(days)]
        return DayPath(name=f"basis_blowout_{carry_shift:+.0%}_{days}d", steps=steps + _hold(total_days, days),
                       description="annualised carry shifts in parallel, then holds")

    @staticmethod
    def grind_up_to_ko(pct_per_day: float, days: int, total_days: int) -> DayPath:
        _check(days, total_days)
        steps = [DayStep(i, [ParameterChange("spot", StressType.PERCENTAGE, pct_per_day)]) for i in range(days)]
        return DayPath(name=f"grind_up_{pct_per_day:.2%}_{days}d", steps=steps + _hold(total_days, days),
                       description="steady rally into the KO barrier")


def stress_set(
    paths: Sequence[DayPath], *, start: StartState, calendar: pd.DatetimeIndex, tenor_grid: np.ndarray
) -> MarketPath:
    """Stack equal-length designed paths into one batch."""
    if not paths:
        raise ValidationError("stress_set needs at least one path")
    singles = [market_path_from_day_path(p, start=start, calendar=calendar, tenor_grid=tenor_grid) for p in paths]
    n_days = {s.n_days for s in singles}
    if len(n_days) != 1:
        raise ValidationError(f"designed paths must have equal length, got {sorted(n_days)}")
    first = singles[0]
    return MarketPath(
        dates=first.dates, spot=np.vstack([s.spot for s in singles]), atm_vol=np.vstack([s.atm_vol for s in singles]),
        rate=np.vstack([s.rate for s in singles]), carry=np.vstack([s.carry for s in singles]),
        tenor_grid=first.tenor_grid, meta={"generator": "designed", "scenario_names": [p.name for p in paths],
                                           "start": first.meta["start"]},
    )
