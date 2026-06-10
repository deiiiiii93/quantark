"""FI stress engine implementation."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

from portfolio.fi.portfolio import FIPortfolio
from stresstest.base import BaseStressEngine, ScenarioRunner
from stresstest.fi.config import FIStressConfig
from stresstest.fi.metrics import FIMetricsCalculator
from stresstest.fi.results import FIStressResults, FIScenarioResult
from stresstest.scenario.scenario import Scenario
from stresstest.stress.stress_applicator import StressApplicator
from util.exceptions import ValidationError
from util.numerical import pnl_pct_of_abs_baseline


class FIStressEngine(BaseStressEngine, ScenarioRunner):
    """Stress engine tailored for fixed income portfolios."""

    def __init__(
        self,
        config: Optional[FIStressConfig] = None,
        metrics_calculator: Optional[FIMetricsCalculator] = None,
    ):
        self.config = config or FIStressConfig()
        self.metrics_calculator = metrics_calculator or FIMetricsCalculator(self.config)

    def supports_portfolio(self, portfolio: Any) -> bool:
        return isinstance(portfolio, FIPortfolio)

    def run_static_scenarios(
        self,
        portfolio: FIPortfolio,
        scenarios: Sequence[Scenario],
        baseline_label: str = "Current Market",
    ) -> FIStressResults:
        if not self.supports_portfolio(portfolio):
            raise ValidationError("FIStressEngine requires an FIPortfolio instance")

        if not scenarios:
            raise ValidationError("At least one scenario is required")

        if len(portfolio) == 0:
            raise ValidationError("Portfolio must contain at least one position")

        baseline_value = portfolio.get_portfolio_value()
        baseline_metrics = self.metrics_calculator.portfolio_metrics(portfolio)

        scenario_results: List[FIScenarioResult] = []
        start = time.time()
        for index, scenario in enumerate(scenarios, 1):
            print(f"Running FI scenario {index}/{len(scenarios)}: {scenario.name}")
            scenario_results.append(
                self.evaluate_scenario(portfolio, scenario, baseline_value)
            )

        total_time = time.time() - start

        return FIStressResults(
            baseline_value=baseline_value,
            baseline_greeks=None,
            scenario_results=scenario_results,
            total_execution_time=total_time,
            config_summary=self.config.get_summary(),
            metadata={"baseline_label": baseline_label},
            extra_metrics={"fi": baseline_metrics, "hedge": self.config.hedge_metadata},
        )

    def evaluate_scenario(
        self,
        portfolio: FIPortfolio,
        scenario: Scenario,
        baseline_value: float,
    ) -> FIScenarioResult:
        scenario_start = time.time()
        stressed_envs = StressApplicator.apply_scenario_to_portfolio(portfolio, scenario)
        stressed_portfolio = self._create_stressed_portfolio(portfolio, stressed_envs)

        stressed_value = stressed_portfolio.get_portfolio_value()
        portfolio_pnl = stressed_value - baseline_value
        portfolio_pnl_pct = pnl_pct_of_abs_baseline(portfolio_pnl, baseline_value)

        fi_metrics = self.metrics_calculator.portfolio_metrics(stressed_portfolio)
        fi_metrics["alerts"] = {
            "dv01_exceeds": abs(fi_metrics.get("dv01", 0.0))
            > self.config.dv01_alert_threshold,
            "convexity_exceeds": abs(fi_metrics.get("convexity", 0.0))
            > self.config.convexity_alert_threshold,
        }
        fi_metrics["curve_summary"] = self._build_curve_summary(
            portfolio, stressed_portfolio
        )

        position_results: List[Dict[str, Any]] = []
        if self.config.save_detailed_results:
            position_results = self._calculate_position_results(
                portfolio,
                stressed_portfolio,
            )

        underlying_results = self._calculate_underlying_results(stressed_portfolio)

        execution_time = time.time() - scenario_start
        return FIScenarioResult(
            scenario=scenario,
            portfolio_value=stressed_value,
            portfolio_pnl=portfolio_pnl,
            portfolio_pnl_pct=portfolio_pnl_pct,
            position_results=position_results,
            underlying_results=underlying_results,
            execution_time=execution_time,
            extra_metrics={"fi": fi_metrics},
        )

    def run_dynamic_scenarios(
        self,
        portfolio: FIPortfolio,
        scenarios: Sequence[Scenario],
        time_steps: Sequence[Any],
        hedge_strategy: Optional[Any] = None,
    ) -> FIStressResults:
        raise NotImplementedError("Dynamic FI scenarios are not implemented yet.")

    def _create_stressed_portfolio(
        self,
        original_portfolio: FIPortfolio,
        stressed_envs: Dict[str, Any],
    ) -> FIPortfolio:
        stressed_portfolio = FIPortfolio(
            portfolio_name=original_portfolio.portfolio_name + "_stressed",
            pricing_environments=stressed_envs,
            creation_date=original_portfolio.creation_date,
        )
        stressed_portfolio.positions = deepcopy(original_portfolio.positions)
        return stressed_portfolio

    def _calculate_position_results(
        self,
        original_portfolio: FIPortfolio,
        stressed_portfolio: FIPortfolio,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
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
                "dv01": position.get_dv01(stressed_env),
                "modified_duration": position.get_modified_duration(stressed_env),
                "convexity": position.get_convexity(stressed_env),
            }
            results.append(result)
        return results

    def _calculate_underlying_results(
        self,
        stressed_portfolio: FIPortfolio,
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for underlying, env in stressed_portfolio.pricing_environments.items():

            positions = stressed_portfolio.get_positions_by_underlying(underlying)
            if not positions:
                continue

            agg_market_value = 0.0
            agg_dv01 = 0.0
            agg_convexity = 0.0
            weighted_duration = 0.0

            for position in positions:
                mv = position.get_market_value(env)
                agg_market_value += mv
                agg_dv01 += position.get_dv01(env)
                agg_convexity += position.get_convexity(env)
                weighted_duration += mv * position.get_modified_duration(env)

            duration = (weighted_duration / agg_market_value) if agg_market_value else 0.0
            results[underlying] = {
                "num_positions": len(positions),
                "total_value": agg_market_value,
                "dv01": agg_dv01,
                "convexity": agg_convexity,
                "modified_duration": duration,
            }
        return results

    def _build_curve_summary(
        self, original: FIPortfolio, stressed: FIPortfolio
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for underlying, env in original.pricing_environments.items():
            stressed_env = stressed.pricing_environments.get(underlying)
            if stressed_env:
                summary[underlying] = StressApplicator.get_stress_summary(env, stressed_env)
        return summary
