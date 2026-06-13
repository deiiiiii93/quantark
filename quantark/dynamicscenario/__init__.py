"""
Dynamic Scenario Analysis module for multi-day stress testing with hedging.

This module provides tools for simulating portfolio evolution over multi-day
market scenarios with optional hedging strategies. Supports both equity and
fixed income portfolios.

Key components:
- DayPath: Defines market parameter evolution over days
- PathBuilder: Fluent API for constructing day paths
- PathLibrary: Predefined path patterns (rally, decline, etc.)
- FIPathLibrary: FI-specific path patterns (rate hikes, curve twists, etc.)
- DynamicScenarioEngine: Main engine for running simulations (equity)
- FIDynamicScenarioEngine: FI-specific engine
- DynamicScenarioResults: Results with day-by-day evolution
- FIDynamicScenarioResults: FI-specific results with DV01/duration tracking
"""

# Base protocols
from quantark.dynamicscenario.base import (
    BaseDynamicScenarioEngine,
    BaseScenarioResults,
    BaseDayResult,
    RiskMetricsAdapter,
    get_engine_for_portfolio,
)

# Path components
from quantark.dynamicscenario.path.day_path import DayStep, DayPath, ParameterChange
from quantark.dynamicscenario.path.path_builder import PathBuilder
from quantark.dynamicscenario.path.path_library import PathLibrary

# Equity (default) components
from quantark.dynamicscenario.config import DynamicScenarioConfig
from quantark.dynamicscenario.engine import DynamicScenarioEngine
from quantark.dynamicscenario.results.dynamic_results import (
    DayResult,
    DynamicScenarioResults,
    PositionSnapshot,
    TradeSnapshot,
    MarketState,
    LifecycleEventSnapshot,
)
from quantark.dynamicscenario.results.result_exporter import DynamicResultExporter
from quantark.dynamicscenario.report.dynamic_report import DynamicReportGenerator
from quantark.dynamicscenario.lifecycle_manager import LifecycleManager
from quantark.dynamicscenario.report.visualizer import DynamicScenarioVisualizer

# FI path library
from quantark.dynamicscenario.path.fi_path_library import FIPathLibrary

# FI components
from quantark.dynamicscenario.fi import (
    FIDynamicScenarioConfig,
    FIDynamicScenarioEngine,
    FIDayResult,
    FIDynamicScenarioResults,
    FIMarketState,
    FITradeSnapshot,
)

# FX components
from quantark.dynamicscenario.fx import (
    FXDynamicScenarioConfig,
    FXDynamicScenarioEngine,
    FXRiskMetricsAdapter,
    FXPathLibrary,
)

# Credit components
from quantark.dynamicscenario.credit import (
    CreditDynamicScenarioConfig,
    CreditDynamicScenarioEngine,
    CreditRiskMetricsAdapter,
    CreditPathLibrary,
)

__all__ = [
    # Base protocols
    "BaseDynamicScenarioEngine",
    "BaseScenarioResults",
    "BaseDayResult",
    "RiskMetricsAdapter",
    "get_engine_for_portfolio",
    # Path components
    "DayStep",
    "DayPath",
    "ParameterChange",
    "PathBuilder",
    "PathLibrary",
    "FIPathLibrary",
    # Equity config
    "DynamicScenarioConfig",
    # Equity engine
    "DynamicScenarioEngine",
    # Equity results
    "DayResult",
    "DynamicScenarioResults",
    "PositionSnapshot",
    "TradeSnapshot",
    "MarketState",
    "DynamicResultExporter",
    # Lifecycle
    "LifecycleManager",
    "LifecycleEventSnapshot",
    # Report & Visualization
    "DynamicReportGenerator",
    "DynamicScenarioVisualizer",
    # FI config
    "FIDynamicScenarioConfig",
    # FI engine
    "FIDynamicScenarioEngine",
    # FI results
    "FIDayResult",
    "FIDynamicScenarioResults",
    "FIMarketState",
    "FITradeSnapshot",
    # FX
    "FXDynamicScenarioConfig",
    "FXDynamicScenarioEngine",
    "FXRiskMetricsAdapter",
    "FXPathLibrary",
    # Credit
    "CreditDynamicScenarioConfig",
    "CreditDynamicScenarioEngine",
    "CreditRiskMetricsAdapter",
    "CreditPathLibrary",
]
