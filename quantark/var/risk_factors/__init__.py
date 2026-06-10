"""
Risk factors module.
"""

from var.risk_factors.base import RiskFactor
from var.risk_factors.equity_factors import (
    DivYieldShiftFactor,
    RateShiftFactor,
    SpotReturnFactor,
    VolChangeFactor,
)
from var.risk_factors.fi_factors import KeyRateShiftFactor, ParallelShiftFactor

__all__ = [
    "RiskFactor",
    "SpotReturnFactor",
    "VolChangeFactor",
    "RateShiftFactor",
    "DivYieldShiftFactor",
    "ParallelShiftFactor",
    "KeyRateShiftFactor",
]
