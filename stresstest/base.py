"""
Core protocols and data contracts shared across stress testing engines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from portfolio.base import BasePortfolio
from stresstest.scenario.scenario import Scenario


@dataclass
class ScenarioEnvelope:
    """
    Standard container for scenario-level stress results.
    """

    scenario: Scenario
    portfolio_value: float
    portfolio_pnl: float
    portfolio_pnl_pct: float
    greeks: Optional[Dict[str, float]] = None
    position_results: List[Dict[str, Any]] = field(default_factory=list)
    underlying_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extra_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressResultEnvelope:
    """
    Top-level container returned by every stress engine implementation.
    """

    baseline_value: float
    baseline_greeks: Optional[Dict[str, float]]
    scenario_results: Sequence[ScenarioEnvelope]
    execution_timestamp: datetime = field(default_factory=datetime.now)
    total_execution_time: float = 0.0
    config_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@runtime_checkable
class StressMetricsAdapter(Protocol):
    """
    Adapter interface for calculating asset-specific stress metrics.
    """

    def supports(self, portfolio: BasePortfolio) -> bool:
        ...

    def compute_metrics(
        self,
        original_portfolio: BasePortfolio,
        stressed_portfolio: BasePortfolio,
        scenario: Scenario,
        baseline_value: float,
        stressed_value: float,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return asset-specific metrics keyed by namespace (e.g. 'equity', 'fi').
        """
        ...


@runtime_checkable
class ScenarioRunner(Protocol):
    """
    Contract for scenario evaluation helpers used by stress engines.
    """

    def evaluate_scenario(
        self,
        portfolio: BasePortfolio,
        scenario: Scenario,
        baseline_value: float,
    ) -> ScenarioEnvelope:
        ...


@runtime_checkable
class BaseStressEngine(Protocol):
    """
    Base protocol implemented by all stress testing engines.
    """

    config: Any

    def supports_portfolio(self, portfolio: BasePortfolio) -> bool:
        """
        Whether this engine can evaluate the supplied portfolio type.
        """
        ...

    def run_static_scenarios(
        self,
        portfolio: BasePortfolio,
        scenarios: Sequence[Scenario],
        baseline_label: str = "Current Market",
    ) -> StressResultEnvelope:
        ...

    def evaluate_scenario(
        self,
        portfolio: BasePortfolio,
        scenario: Scenario,
        baseline_value: float,
    ) -> ScenarioEnvelope:
        ...


