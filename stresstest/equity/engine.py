"""Equity stress engine implementation."""

from __future__ import annotations

import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from asset.equity.riskmeasures import GreeksCalculator
from portfolio import Portfolio
from portfolio.base import BasePortfolio
from stresstest.base import BaseStressEngine, ScenarioRunner, StressMetricsAdapter
from stresstest.equity.config import EquityStressConfig
from stresstest.equity.results import ScenarioResult, StressTestResults
from stresstest.scenario.scenario import Scenario
from stresstest.stress.stress_applicator import StressApplicator
from util.exceptions import ValidationError
from util.numerical import pnl_pct_of_abs_baseline


class EquityStressMetricsAdapter(StressMetricsAdapter):
    """Default adapter that can be extended for equity-specific metrics."""

    def supports(self, portfolio: BasePortfolio) -> bool:
        return True

    def compute_metrics(
        self,
        original_portfolio: BasePortfolio,
        stressed_portfolio: BasePortfolio,
        scenario: Scenario,
        baseline_value: float,
        stressed_value: float,
    ) -> Dict[str, Dict[str, Any]]:
        return {}


class EquityStressEngine(BaseStressEngine, ScenarioRunner):
    """
    Engine for executing stress tests on portfolios.
    """

    def __init__(
        self,
        config: Optional[EquityStressConfig] = None,
        metrics_adapter: Optional[StressMetricsAdapter] = None,
    ):
        self.config = config or EquityStressConfig()
        self.metrics_adapter = metrics_adapter or EquityStressMetricsAdapter()
        self.greeks_calculator = (
            GreeksCalculator() if self.config.calculate_greeks else None
        )

    def supports_portfolio(self, portfolio: BasePortfolio) -> bool:
        required_attrs = [
            "get_portfolio_value",
            "get_portfolio_greeks",
            "pricing_environments",
        ]
        return all(hasattr(portfolio, attr) for attr in required_attrs)

    def run_static_scenarios(
        self,
        portfolio: Portfolio,
        scenarios: Sequence[Scenario],
        baseline_label: str = "Current Market",
    ) -> StressTestResults:
        if not portfolio or len(portfolio) == 0:
            raise ValidationError("Portfolio must contain at least one position")

        if not scenarios:
            raise ValidationError("At least one scenario is required")

        start_time = time.time()

        print("Evaluating baseline portfolio...")
        baseline_value = portfolio.get_portfolio_value()
        baseline_greeks = None

        if self.config.calculate_greeks and self.greeks_calculator:
            use_analytical = self.config.greeks_method == "analytical"
            baseline_greeks = portfolio.get_portfolio_greeks(
                self.greeks_calculator,
                use_analytical=use_analytical,
            )

        scenario_results: List[ScenarioResult] = []
        for i, scenario in enumerate(scenarios, 1):
            print(f"Running scenario {i}/{len(scenarios)}: {scenario.name}")
            result = self.evaluate_scenario(portfolio, scenario, baseline_value)
            scenario_results.append(result)

        total_time = time.time() - start_time

        return StressTestResults(
            baseline_value=baseline_value,
            baseline_greeks=baseline_greeks,
            scenario_results=scenario_results,
            total_execution_time=total_time,
            config_summary=self.config.get_summary(),
            metadata={"baseline_label": baseline_label},
        )

    def evaluate_scenario(
        self,
        portfolio: Portfolio,
        scenario: Scenario,
        baseline_value: float,
    ) -> ScenarioResult:
        scenario_start = time.time()

        stressed_envs = StressApplicator.apply_scenario_to_portfolio(portfolio, scenario)
        stressed_portfolio = self._create_stressed_portfolio(portfolio, stressed_envs)

        stressed_value = stressed_portfolio.get_portfolio_value()
        portfolio_pnl = stressed_value - baseline_value
        portfolio_pnl_pct = pnl_pct_of_abs_baseline(portfolio_pnl, baseline_value)

        greeks = None
        if self.config.calculate_greeks and self.greeks_calculator:
            use_analytical = self.config.greeks_method == "analytical"
            greeks = stressed_portfolio.get_portfolio_greeks(
                self.greeks_calculator,
                use_analytical=use_analytical,
            )

        position_results: List[Dict[str, Any]] = []
        if self.config.save_detailed_results:
            position_results = self._calculate_position_results(
                portfolio,
                stressed_portfolio,
                scenario,
            )

        underlying_results = self._calculate_underlying_results(
            stressed_portfolio,
            scenario,
        )

        extra_metrics: Dict[str, Dict[str, Any]] = {}
        if self.metrics_adapter and self.metrics_adapter.supports(portfolio):
            extra_metrics = self.metrics_adapter.compute_metrics(
                portfolio,
                stressed_portfolio,
                scenario,
                baseline_value,
                stressed_value,
            )

        execution_time = time.time() - scenario_start

        return ScenarioResult(
            scenario=scenario,
            portfolio_value=stressed_value,
            portfolio_pnl=portfolio_pnl,
            portfolio_pnl_pct=portfolio_pnl_pct,
            greeks=greeks,
            position_results=position_results,
            underlying_results=underlying_results,
            execution_time=execution_time,
            extra_metrics=extra_metrics,
        )

    def run_dynamic_scenarios(
        self,
        portfolio: Portfolio,
        scenarios: Sequence[Scenario],
        time_steps: Sequence[datetime],
        hedge_strategy: Optional[Any] = None,
    ) -> StressTestResults:
        raise NotImplementedError(
            "Dynamic scenario analysis is not yet implemented. "
            "This API is reserved for future development."
        )

    def _create_stressed_portfolio(
        self,
        original_portfolio: Portfolio,
        stressed_envs: Dict[str, Any],
    ) -> Portfolio:
        stressed_portfolio = Portfolio(
            portfolio_name=original_portfolio.portfolio_name + "_stressed",
            pricing_environments=stressed_envs,
            creation_date=original_portfolio.creation_date,
        )
        stressed_portfolio.positions = deepcopy(original_portfolio.positions)
        return stressed_portfolio

    def _calculate_position_results(
        self,
        original_portfolio: Portfolio,
        stressed_portfolio: Portfolio,
        scenario: Scenario,
    ) -> List[Dict[str, Any]]:
        results = []

        for position_id, position in stressed_portfolio.positions.items():
            original_position = original_portfolio.positions[position_id]
            original_env = original_portfolio.pricing_environments[position.underlying]
            stressed_env = stressed_portfolio.pricing_environments[position.underlying]

            original_value = original_position.get_market_value(original_env)
            stressed_value = position.get_market_value(stressed_env)

            position_pnl = stressed_value - original_value
            position_pnl_pct = pnl_pct_of_abs_baseline(position_pnl, original_value)

            result = {
                "position_id": position_id,
                "underlying": position.underlying,
                "product_type": position.product.__class__.__name__,
                "quantity": position.quantity,
                "original_value": original_value,
                "stressed_value": stressed_value,
                "pnl": position_pnl,
                "pnl_pct": position_pnl_pct,
            }

            if self.config.calculate_greeks and self.greeks_calculator:
                use_analytical = self.config.greeks_method == "analytical"
                position_greeks = position.get_greeks(
                    stressed_env,
                    self.greeks_calculator,
                    use_analytical=use_analytical,
                )
                result["greeks"] = position_greeks

            results.append(result)

        return results

    def _calculate_underlying_results(
        self,
        stressed_portfolio: Portfolio,
        scenario: Scenario,
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}

        for underlying in stressed_portfolio.pricing_environments.keys():
            positions = stressed_portfolio.get_positions_by_underlying(underlying)
            if not positions:
                continue

            total_value = sum(
                pos.get_market_value(stressed_portfolio.pricing_environments[underlying])
                for pos in positions
            )

            greeks = None
            if self.config.calculate_greeks and self.greeks_calculator:
                use_analytical = self.config.greeks_method == "analytical"
                greeks = stressed_portfolio.get_greeks_by_underlying(
                    underlying,
                    self.greeks_calculator,
                    use_analytical=use_analytical,
                )

            results[underlying] = {
                "num_positions": len(positions),
                "total_value": total_value,
                "greeks": greeks,
            }

        return results

    def __repr__(self) -> str:
        return f"EquityStressEngine(config={self.config})"


# Backward-compatible alias to avoid breaking external imports.
StressTestEngine = EquityStressEngine
