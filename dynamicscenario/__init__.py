"""
Dynamic Scenario Analysis module for multi-day stress testing with hedging.

This module provides tools for simulating portfolio evolution over multi-day
market scenarios with optional hedging strategies.

Key components:
- DayPath: Defines market parameter evolution over days
- PathBuilder: Fluent API for constructing day paths
- PathLibrary: Predefined path patterns (rally, decline, etc.)
- DynamicScenarioEngine: Main engine for running simulations
- DynamicScenarioResults: Results with day-by-day evolution
"""

from dynamicscenario.path.day_path import DayStep, DayPath, ParameterChange
from dynamicscenario.path.path_builder import PathBuilder
from dynamicscenario.path.path_library import PathLibrary
from dynamicscenario.config import DynamicScenarioConfig
from dynamicscenario.engine import DynamicScenarioEngine
from dynamicscenario.results.dynamic_results import (
    DayResult, DynamicScenarioResults, PositionSnapshot,
    TradeSnapshot, MarketState
)
from dynamicscenario.results.result_exporter import DynamicResultExporter
from dynamicscenario.report.dynamic_report import DynamicReportGenerator
from dynamicscenario.report.visualizer import DynamicScenarioVisualizer

__all__ = [
    # Path components
    'DayStep',
    'DayPath',
    'ParameterChange',
    'PathBuilder',
    'PathLibrary',
    # Config
    'DynamicScenarioConfig',
    # Engine
    'DynamicScenarioEngine',
    # Results
    'DayResult',
    'DynamicScenarioResults',
    'PositionSnapshot',
    'TradeSnapshot',
    'MarketState',
    'DynamicResultExporter',
    # Report & Visualization
    'DynamicReportGenerator',
    'DynamicScenarioVisualizer',
]

