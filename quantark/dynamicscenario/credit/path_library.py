"""
Predefined credit day-path patterns for dynamic scenario analysis.

Builds DayPath objects using the credit parameter set (spread/hazard, rate).
Spread and rate changes are expressed in basis points per day and applied as
absolute shifts.
"""
from __future__ import annotations

from typing import Optional

from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.stresstest.stress.stress_types import StressLevel, StressType

_BPS = 1e-4


def _level_target(entity: Optional[str]):
    if entity:
        return StressLevel.UNDERLYING, entity
    return StressLevel.PORTFOLIO, None


class CreditPathLibrary:
    """Factory of common credit multi-day paths."""

    @staticmethod
    def spread_widening(days: int = 5, bps_per_day: float = 10.0,
                        entity: Optional[str] = None,
                        name: Optional[str] = None) -> DayPath:
        """Steady credit-spread (hazard) widening, bps/day."""
        level, target = _level_target(entity)
        steps = [
            DayStep(day_index=i, changes=[
                ParameterChange("spread", StressType.ABSOLUTE, bps_per_day * _BPS,
                                level, target)
            ], label=f"Day {i + 1}")
            for i in range(days)
        ]
        return DayPath(name=name or f"Spread Widening {bps_per_day:.0f}bp/day",
                       steps=steps, description="Steady credit-spread widening")

    @staticmethod
    def spread_rally(days: int = 5, bps_per_day: float = 8.0,
                     entity: Optional[str] = None) -> DayPath:
        """Steady credit-spread tightening, bps/day."""
        level, target = _level_target(entity)
        steps = [
            DayStep(day_index=i, changes=[
                ParameterChange("spread", StressType.ABSOLUTE, -bps_per_day * _BPS,
                                level, target)
            ], label=f"Day {i + 1}")
            for i in range(days)
        ]
        return DayPath(name="Spread Rally", steps=steps,
                       description="Steady credit-spread tightening")

    @staticmethod
    def credit_crisis(days: int = 6, spike_bps: float = 50.0, decay_bps: float = 8.0,
                      rate_flight_bps: float = 5.0,
                      entity: Optional[str] = None) -> DayPath:
        """Day-1 spread blowout with flight-to-quality rate rally, then decay."""
        level, target = _level_target(entity)
        steps = []
        for i in range(days):
            spread_val = spike_bps if i == 0 else -decay_bps
            changes = [ParameterChange("spread", StressType.ABSOLUTE, spread_val * _BPS,
                                       level, target)]
            if i == 0 and rate_flight_bps:
                changes.append(ParameterChange("rate", StressType.ABSOLUTE,
                                               -rate_flight_bps * _BPS, level, target))
            steps.append(DayStep(day_index=i, changes=changes,
                                 label="Blowout" if i == 0 else f"Recovery {i}"))
        return DayPath(name="Credit Crisis", steps=steps,
                       description="Spread blowout with flight-to-quality, then decay")
