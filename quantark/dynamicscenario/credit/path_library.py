"""
Predefined credit day-path patterns for dynamic scenario analysis.

Builds DayPath objects on the canonical credit risk factor, the hazard
intensity (plus the rate factor). Hazard and rate changes are expressed in basis
points per day and applied as absolute shifts. The ``spread_*`` factory names
are retained as deprecated aliases of the ``hazard_*`` paths: a curve-level move
is a hazard-intensity move, not a recovery-dependent quoted-spread move.
"""
from __future__ import annotations

import warnings
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
    def hazard_widening(days: int = 5, bps_per_day: float = 10.0,
                        entity: Optional[str] = None,
                        name: Optional[str] = None) -> DayPath:
        """Steady hazard-intensity widening, bps/day."""
        level, target = _level_target(entity)
        steps = [
            DayStep(day_index=i, changes=[
                ParameterChange("hazard", StressType.ABSOLUTE, bps_per_day * _BPS,
                                level, target)
            ], label=f"Day {i + 1}")
            for i in range(days)
        ]
        return DayPath(name=name or f"Hazard Widening {bps_per_day:.0f}bp/day",
                       steps=steps, description="Steady hazard-intensity widening")

    @staticmethod
    def hazard_rally(days: int = 5, bps_per_day: float = 8.0,
                     entity: Optional[str] = None) -> DayPath:
        """Steady hazard-intensity tightening, bps/day."""
        level, target = _level_target(entity)
        steps = [
            DayStep(day_index=i, changes=[
                ParameterChange("hazard", StressType.ABSOLUTE, -bps_per_day * _BPS,
                                level, target)
            ], label=f"Day {i + 1}")
            for i in range(days)
        ]
        return DayPath(name="Hazard Rally", steps=steps,
                       description="Steady hazard-intensity tightening")

    @staticmethod
    def spread_widening(days: int = 5, bps_per_day: float = 10.0,
                        entity: Optional[str] = None,
                        name: Optional[str] = None) -> DayPath:
        """Deprecated alias of :meth:`hazard_widening` (a hazard-intensity path)."""
        warnings.warn(
            "CreditPathLibrary.spread_widening is deprecated; a curve-level move "
            "is a hazard-intensity move. Use hazard_widening.",
            DeprecationWarning, stacklevel=2,
        )
        return CreditPathLibrary.hazard_widening(days, bps_per_day, entity, name)

    @staticmethod
    def spread_rally(days: int = 5, bps_per_day: float = 8.0,
                     entity: Optional[str] = None) -> DayPath:
        """Deprecated alias of :meth:`hazard_rally` (a hazard-intensity path)."""
        warnings.warn(
            "CreditPathLibrary.spread_rally is deprecated; a curve-level move is a "
            "hazard-intensity move. Use hazard_rally.",
            DeprecationWarning, stacklevel=2,
        )
        return CreditPathLibrary.hazard_rally(days, bps_per_day, entity)

    @staticmethod
    def credit_crisis(days: int = 6, spike_bps: float = 50.0, decay_bps: float = 8.0,
                      rate_flight_bps: float = 5.0,
                      entity: Optional[str] = None) -> DayPath:
        """Day-1 hazard blowout with flight-to-quality rate rally, then decay."""
        level, target = _level_target(entity)
        steps = []
        for i in range(days):
            hazard_val = spike_bps if i == 0 else -decay_bps
            changes = [ParameterChange("hazard", StressType.ABSOLUTE, hazard_val * _BPS,
                                       level, target)]
            if i == 0 and rate_flight_bps:
                changes.append(ParameterChange("rate", StressType.ABSOLUTE,
                                               -rate_flight_bps * _BPS, level, target))
            steps.append(DayStep(day_index=i, changes=changes,
                                 label="Blowout" if i == 0 else f"Recovery {i}"))
        return DayPath(name="Credit Crisis", steps=steps,
                       description="Hazard blowout with flight-to-quality, then decay")
