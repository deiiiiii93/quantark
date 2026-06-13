"""Credit stress engine implementation."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

from quantark.asset.credit.riskmeasures import CreditGreeksCalculator
from quantark.portfolio.credit import CreditPortfolio
from quantark.stresstest.base import BaseStressEngine, ScenarioRunner, StressMetricsAdapter
from quantark.stresstest.equity.results import ScenarioResult, StressTestResults
from quantark.stresstest.credit.config import CreditStressConfig
from quantark.stresstest.credit.credit_stress_applicator import CreditStressApplicator
from quantark.stresstest.scenario.scenario import Scenario
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import pnl_pct_of_abs_baseline


class CreditStressMetricsAdapter(StressMetricsAdapter):
    """Default credit metrics adapter (extension point for bespoke metrics)."""

    def supports(self, portfolio: Any) -> bool:
        return isinstance(portfolio, CreditPortfolio)

    def compute_metrics(
        self,
        original_portfolio: Any,
        stressed_portfolio: Any,
        scenario: Scenario,
        baseline_value: float,
        stressed_value: float,
    ) -> Dict[str, Dict[str, Any]]:
        return {}


class CreditStressEngine(BaseStressEngine, ScenarioRunner):
    """Stress engine for credit portfolios (spread / rate shocks)."""

    def __init__(
        self,
        config: Optional[CreditStressConfig] = None,
        metrics_adapter: Optional[StressMetricsAdapter] = None,
    ):
        self.config = config or CreditStressConfig()
        self.metrics_adapter = metrics_adapter or CreditStressMetricsAdapter()
        self.greeks_calculator = (
            CreditGreeksCalculator() if self.config.calculate_greeks else None
        )

    def supports_portfolio(self, portfolio: Any) -> bool:
        required = ["get_portfolio_value", "get_portfolio_greeks", "pricing_environments"]
        return all(hasattr(portfolio, attr) for attr in required)

    def run_static_scenarios(
        self,
        portfolio: CreditPortfolio,
        scenarios: Sequence[Scenario],
        baseline_label: str = "Current Market",
    ) -> StressTestResults:
        if not portfolio or len(portfolio) == 0:
            raise ValidationError("Portfolio must contain at least one position")
        if not scenarios:
            raise ValidationError("At least one scenario is required")

        start_time = time.time()
        baseline_value = portfolio.get_portfolio_value()
        baseline_greeks = None
        if self.greeks_calculator is not None:
            baseline_greeks = portfolio.get_portfolio_greeks(self.greeks_calculator)

        scenario_results: List[ScenarioResult] = [
            self.evaluate_scenario(portfolio, scenario, baseline_value)
            for scenario in scenarios
        ]

        return StressTestResults(
            baseline_value=baseline_value,
            baseline_greeks=baseline_greeks,
            scenario_results=scenario_results,
            total_execution_time=time.time() - start_time,
            config_summary=self.config.get_summary(),
            metadata={"baseline_label": baseline_label, "asset_class": "credit"},
        )

    def evaluate_scenario(
        self,
        portfolio: CreditPortfolio,
        scenario: Scenario,
        baseline_value: float,
    ) -> ScenarioResult:
        scenario_start = time.time()

        stressed_envs = CreditStressApplicator.apply_scenario_to_portfolio(
            portfolio, scenario
        )
        stressed_portfolio = self._create_stressed_portfolio(portfolio, stressed_envs)

        stressed_value = stressed_portfolio.get_portfolio_value()
        portfolio_pnl = stressed_value - baseline_value
        portfolio_pnl_pct = pnl_pct_of_abs_baseline(portfolio_pnl, baseline_value)

        greeks = None
        if self.greeks_calculator is not None:
            greeks = stressed_portfolio.get_portfolio_greeks(self.greeks_calculator)

        position_results: List[Dict[str, Any]] = []
        if self.config.save_detailed_results:
            position_results = self._calculate_position_results(
                portfolio, stressed_portfolio
            )

        underlying_results = self._calculate_underlying_results(stressed_portfolio)

        extra_metrics: Dict[str, Dict[str, Any]] = {}
        if self.metrics_adapter and self.metrics_adapter.supports(portfolio):
            extra_metrics = self.metrics_adapter.compute_metrics(
                portfolio, stressed_portfolio, scenario, baseline_value, stressed_value
            )

        return ScenarioResult(
            scenario=scenario,
            portfolio_value=stressed_value,
            portfolio_pnl=portfolio_pnl,
            portfolio_pnl_pct=portfolio_pnl_pct,
            greeks=greeks,
            position_results=position_results,
            underlying_results=underlying_results,
            execution_time=time.time() - scenario_start,
            extra_metrics=extra_metrics,
        )

    # ------------------------------------------------------------------ #
    def _create_stressed_portfolio(
        self,
        original: CreditPortfolio,
        stressed_envs: Dict[str, Any],
    ) -> CreditPortfolio:
        stressed = CreditPortfolio(
            portfolio_name=original.portfolio_name + "_stressed",
            pricing_environments=stressed_envs,
            creation_date=original.creation_date,
        )
        stressed.positions = deepcopy(original.positions)
        return stressed

    def _calculate_position_results(
        self,
        original: CreditPortfolio,
        stressed: CreditPortfolio,
    ) -> List[Dict[str, Any]]:
        results = []
        for position_id, position in stressed.positions.items():
            original_position = original.positions[position_id]
            entity = position.reference_entity
            original_env = original.pricing_environments[entity]
            stressed_env = stressed.pricing_environments[entity]

            original_value = original_position.get_market_value(original_env)
            stressed_value = position.get_market_value(stressed_env)
            position_pnl = stressed_value - original_value

            row = {
                "position_id": position_id,
                "reference_entity": entity,
                "product_type": position.product.__class__.__name__,
                "quantity": position.quantity,
                "original_value": original_value,
                "stressed_value": stressed_value,
                "pnl": position_pnl,
                "pnl_pct": pnl_pct_of_abs_baseline(position_pnl, original_value),
            }
            if self.greeks_calculator is not None:
                row["greeks"] = position.get_greeks(stressed_env, self.greeks_calculator)
            results.append(row)
        return results

    def _calculate_underlying_results(
        self, stressed: CreditPortfolio
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for entity in stressed.pricing_environments:
            positions = stressed.get_positions_by_underlying(entity)
            if not positions:
                continue
            env = stressed.pricing_environments[entity]
            total_value = sum(pos.get_market_value(env) for pos in positions)
            greeks = None
            if self.greeks_calculator is not None:
                greeks = stressed.get_greeks_by_underlying(entity, self.greeks_calculator)
            results[entity] = {
                "num_positions": len(positions),
                "total_value": total_value,
                "greeks": greeks,
            }
        return results

    def __repr__(self) -> str:
        return f"CreditStressEngine(config={self.config})"
