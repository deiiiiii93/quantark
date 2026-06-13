"""
Stress Test Module for Portfolio Scenario Analysis.

This module provides comprehensive stress testing capabilities for portfolios,
allowing users to analyze how different market conditions affect portfolio P&L
and risk exposures.

Key Features:
- Flexible parameter stressing at portfolio, underlying, or position level
- Predefined common scenarios and custom scenario definition
- YAML/JSON scenario storage for reusability
- Multi-parameter simultaneous stress capability
- Comprehensive reporting with HTML, CSV, Parquet, and visualizations
- API design ready for future dynamic scenario analysis

Example:
    >>> from stresstest import StressTestEngine, StressTestConfig, ScenarioBuilder
    >>> from stresstest.scenario import ScenarioLibrary
    >>> 
    >>> # Create stress test configuration
    >>> config = StressTestConfig(
    ...     calculate_greeks=True,
    ...     export_format=['parquet', 'csv']
    ... )
    >>> 
    >>> # Build or load scenarios
    >>> scenarios = [
    ...     ScenarioLibrary.market_crash(),
    ...     ScenarioLibrary.vol_spike(),
    ...     ScenarioBuilder().name("Custom").spot_stress(-0.15).build()
    ... ]
    >>> 
    >>> # Run stress test
    >>> engine = StressTestEngine(config)
    >>> results = engine.run_static_scenarios(portfolio, scenarios)
    >>> 
    >>> # Generate reports
    >>> results.to_parquet("stress_results.parquet")
    >>> results.generate_report("stress_report.html")
"""

from quantark.stresstest.config import StressTestConfig
from quantark.stresstest.engine import StressTestEngine
from quantark.stresstest.fi import FIStressConfig, FIStressEngine
from quantark.stresstest.fx import FXStressConfig, FXStressEngine
from quantark.stresstest.scenario.scenario import Scenario, Stress
from quantark.stresstest.scenario.scenario_builder import ScenarioBuilder
from quantark.stresstest.stress.stress_types import (
    StressType,
    StressLevel,
    BasisDividendRelationshipMode,
)

__version__ = "0.1.0"

__all__ = [
    "StressTestConfig",
    "StressTestEngine",
    "FIStressConfig",
    "FIStressEngine",
    "FXStressConfig",
    "FXStressEngine",
    "Scenario",
    "Stress",
    "ScenarioBuilder",
    "StressType",
    "StressLevel",
    "BasisDividendRelationshipMode",
]

