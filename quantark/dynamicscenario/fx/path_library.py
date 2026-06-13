"""
Predefined FX day-path patterns for dynamic scenario analysis.

Builds :class:`DayPath` objects using the FX parameter set (spot, vol,
domestic_rate, foreign_rate). Rate changes are expressed in basis points per
day and applied as absolute curve shifts.
"""
from __future__ import annotations

from typing import Optional

from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.stresstest.stress.stress_types import StressLevel, StressType

_BPS = 1e-4


def _level_target(pair: Optional[str]):
    if pair:
        return StressLevel.UNDERLYING, pair
    return StressLevel.PORTFOLIO, None


class FXPathLibrary:
    """Factory of common FX multi-day paths."""

    @staticmethod
    def spot_trend(days: int = 5, daily_pct: float = 0.01,
                   pair: Optional[str] = None, name: Optional[str] = None) -> DayPath:
        """Constant daily spot drift (e.g. +1%/day)."""
        level, target = _level_target(pair)
        steps = [
            DayStep(day_index=i, changes=[
                ParameterChange("spot", StressType.PERCENTAGE, daily_pct, level, target)
            ], label=f"Day {i + 1}")
            for i in range(days)
        ]
        return DayPath(name=name or f"Spot Trend {daily_pct:+.1%}/day", steps=steps,
                       description="Constant daily FX spot drift")

    @staticmethod
    def rate_divergence(days: int = 5, dom_bps_per_day: float = 5.0,
                        for_bps_per_day: float = -5.0,
                        pair: Optional[str] = None) -> DayPath:
        """Domestic and foreign curves drift apart (carry divergence)."""
        level, target = _level_target(pair)
        steps = []
        for i in range(days):
            changes = []
            if dom_bps_per_day:
                changes.append(ParameterChange(
                    "domestic_rate", StressType.ABSOLUTE, dom_bps_per_day * _BPS,
                    level, target))
            if for_bps_per_day:
                changes.append(ParameterChange(
                    "foreign_rate", StressType.ABSOLUTE, for_bps_per_day * _BPS,
                    level, target))
            steps.append(DayStep(day_index=i, changes=changes, label=f"Day {i + 1}"))
        return DayPath(name="Rate Divergence", steps=steps,
                       description="Domestic up / foreign down carry divergence")

    @staticmethod
    def carry_unwind(days: int = 5, spot_daily_pct: float = -0.01,
                     foreign_bps_per_day: float = 8.0,
                     pair: Optional[str] = None) -> DayPath:
        """Carry-trade unwind: spot sells off while the funded (foreign) rate rises."""
        level, target = _level_target(pair)
        steps = [
            DayStep(day_index=i, changes=[
                ParameterChange("spot", StressType.PERCENTAGE, spot_daily_pct, level, target),
                ParameterChange("foreign_rate", StressType.ABSOLUTE,
                                foreign_bps_per_day * _BPS, level, target),
            ], label=f"Day {i + 1}")
            for i in range(days)
        ]
        return DayPath(name="Carry Unwind", steps=steps,
                       description="Spot sell-off with rising foreign funding rate")

    @staticmethod
    def vol_spike_decay(days: int = 6, spike: float = 0.05, decay: float = 0.01,
                        pair: Optional[str] = None) -> DayPath:
        """Day-1 vol spike (absolute) followed by gradual decay."""
        level, target = _level_target(pair)
        steps = []
        for i in range(days):
            change_val = spike if i == 0 else -decay
            steps.append(DayStep(day_index=i, changes=[
                ParameterChange("vol", StressType.ABSOLUTE, change_val, level, target)
            ], label="Spike" if i == 0 else f"Decay {i}"))
        return DayPath(name="Vol Spike & Decay", steps=steps,
                       description="Volatility spikes then decays")
