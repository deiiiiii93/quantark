"""
Fixed Income dynamic scenario analysis module.

This subpackage contains FI-specific implementations of the dynamic scenario
analysis framework, including the FI engine, config, and results classes.
"""

from dynamicscenario.fi.config import FIDynamicScenarioConfig
from dynamicscenario.fi.engine import FIDynamicScenarioEngine
from dynamicscenario.fi.results import (
    FIDayResult,
    FIDynamicScenarioResults,
    FIMarketState,
    FITradeSnapshot,
)

__all__ = [
    "FIDynamicScenarioConfig",
    "FIDynamicScenarioEngine",
    "FIDayResult",
    "FIDynamicScenarioResults",
    "FIMarketState",
    "FITradeSnapshot",
]
