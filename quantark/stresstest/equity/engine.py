"""Equity stress engine implementation."""

from __future__ import annotations

import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.portfolio import Portfolio
from quantark.portfolio.base import BasePortfolio
from quantark.stresstest.base import BaseStressEngine, ScenarioRunner, StressMetricsAdapter
from quantark.stresstest.equity.config import EquityStressConfig
from quantark.stresstest.equity.results import ScenarioResult, StressTestResults
from quantark.stresstest.equity.parallel import position_task, run_tasks, scenario_task
from quantark.stresstest.equity.valuation import (
    PositionValuation,
    aggregate_greeks,
    supports_native_batch,
    value_positions,
)
from quantark.stresstest.scenario.scenario import Scenario
from quantark.stresstest.stress.stress_applicator import StressApplicator
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import pnl_pct_of_abs_baseline
from quantark.util.fixings import freeze_portfolio_fixings


def _tagged_task(tagged):
    """A pool task carrying its own function (position or scenario task); module level for spawn."""
    fn, task = tagged
    return fn(task)


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
        # A missing past fixing is history: filled once at the base spot, kept under every shock (2026-10-09)
        portfolio = freeze_portfolio_fixings(portfolio)
        if not portfolio or len(portfolio) == 0:
            raise ValidationError("Portfolio must contain at least one position")

        if not scenarios:
            raise ValidationError("At least one scenario is required")

        start_time = time.time()

        print("Evaluating baseline portfolio...")
        # Baseline and per-position originals always take the scalar path, once per run (spec D5, D14)
        baseline = self._value_positions(portfolio.positions, portfolio.pricing_environments)
        baseline_value = sum(v.value for v in baseline.values())
        baseline_greeks = None
        if self.greeks_calculator is not None:
            baseline_greeks = aggregate_greeks(v.greeks for v in baseline.values())
        originals = {position_id: v.value for position_id, v in baseline.items()}

        scenario_results, execution, workers = self._run_scenarios(
            portfolio, scenarios, originals, baseline_value
        )

        total_time = time.time() - start_time

        return StressTestResults(
            baseline_value=baseline_value,
            baseline_greeks=baseline_greeks,
            scenario_results=scenario_results,
            total_execution_time=total_time,
            config_summary=self.config.get_summary(),
            metadata={
                "baseline_label": baseline_label,
                "valuation_mode": self._valuation_mode,
                "execution": execution,
                "workers": workers,
            },
        )

    def evaluate_scenario(
        self,
        portfolio: Portfolio,
        scenario: Scenario,
        baseline_value: float,
    ) -> ScenarioResult:
        # A missing past fixing is history: filled once at the base spot, kept under every shock (2026-10-09)
        portfolio = freeze_portfolio_fixings(portfolio)
        # the originals are priced exactly as a run prices its baseline, so engines reach the same state
        baseline = self._value_positions(portfolio.positions, portfolio.pricing_environments)
        originals = {position_id: v.value for position_id, v in baseline.items()}
        return self._run_scenarios(portfolio, [scenario], originals, baseline_value)[0][0]

    def _run_scenarios(
        self,
        portfolio: Portfolio,
        scenarios: Sequence[Scenario],
        originals: Dict[str, float],
        baseline_value: float,
    ) -> Tuple[List[ScenarioResult], str, int]:
        """The scenario results, how they ran ("serial" | "process_pool") and the worker count."""
        if self.config.native_batch:
            print(f"Running {len(scenarios)} scenarios with native batching")
            return self._run_native_batch(portfolio, scenarios, originals, baseline_value)
        if self._pool_for(len(scenarios)):
            pooled = self._run_scalar_pool(portfolio, scenarios, originals, baseline_value)
            if pooled is not None:
                return pooled
        results = []
        for i, scenario in enumerate(scenarios, 1):
            print(f"Running scenario {i}/{len(scenarios)}: {scenario.name}")
            results.append(self._evaluate_scalar(portfolio, scenario, originals, baseline_value))
        return results, "serial", 0

    def _pool_for(self, n_tasks: int) -> bool:
        return self.config.parallel_execution and self.config.max_workers > 1 and n_tasks > 1

    def _run_scalar_pool(self, portfolio, scenarios, originals, baseline_value):
        """Scalar scenarios as pool tasks: each prices a pickled copy of the post-baseline positions (D13)."""
        start = time.time()
        envs = [StressApplicator.apply_scenario_to_portfolio(portfolio, s) for s in scenarios]
        tasks = [(portfolio.positions, e, self.greeks_calculator, self._use_analytical) for e in envs]
        print(f"Running {len(scenarios)} scenarios on up to {self.config.max_workers} processes")
        pooled = run_tasks(scenario_task, tasks, self.config.max_workers)
        if pooled is None:
            return None
        valuations, workers = pooled
        per_scenario = (time.time() - start) / len(scenarios)
        results = []
        for scenario, scenario_envs, rows in zip(scenarios, envs, valuations):
            result = self._assemble_result(
                portfolio, self._view(portfolio, scenario_envs), scenario, rows, originals, baseline_value
            )
            result.execution_time = per_scenario
            results.append(result)
        return results, "process_pool", workers

    @staticmethod
    def _view(portfolio: Portfolio, envs: Dict[str, Any]) -> Portfolio:
        """The stressed portfolio for assembly: scenario envs, the positions shared (read, never priced)."""
        stressed = Portfolio(
            portfolio_name=portfolio.portfolio_name + "_stressed",
            pricing_environments=envs,
            creation_date=portfolio.creation_date,
        )
        stressed.positions = dict(portfolio.positions)
        return stressed

    def _run_native_batch(
        self,
        portfolio: Portfolio,
        scenarios: Sequence[Scenario],
        originals: Dict[str, float],
        baseline_value: float,
    ) -> List[ScenarioResult]:
        """Every scenario's environments built once; session-capable positions batched across them (spec D14).

        Each session-capable position is valued on its own copy over all scenarios in one spot-batch session;
        every other position is valued exactly as in scalar mode, on a fresh copy per scenario.
        """
        batch_start = time.time()
        envs = [StressApplicator.apply_scenario_to_portfolio(portfolio, s) for s in scenarios]
        batched = [i for i, p in portfolio.positions.items() if supports_native_batch(p)]
        others = {i: p for i, p in portfolio.positions.items() if i not in set(batched)}

        gc = self.greeks_calculator
        position_tasks = [
            (portfolio.positions[i], [e[portfolio.positions[i].underlying] for e in envs], gc) for i in batched
        ]
        scenario_tasks = [(others, e, gc, self._use_analytical) for e in envs] if others else []

        execution, workers, outcome = "serial", 0, None
        if self._pool_for(len(position_tasks) + len(scenario_tasks)):
            tagged = [(position_task, t) for t in position_tasks] + [(scenario_task, t) for t in scenario_tasks]
            pooled = run_tasks(_tagged_task, tagged, self.config.max_workers)
            if pooled is not None:
                outcome, workers = pooled
                execution = "process_pool"
        if outcome is None:
            # serially, each task on its own copy exactly as a worker gets one (spec D14)
            outcome = [position_task(deepcopy(t)) for t in position_tasks]
            outcome += [scenario_task(deepcopy(t)) for t in scenario_tasks]

        valuations: List[Dict[str, PositionValuation]] = [{} for _ in scenarios]
        for position_id, rows in zip(batched, outcome[: len(position_tasks)]):
            for k, row in enumerate(rows):
                valuations[k][position_id] = row
        for k, rows in enumerate(outcome[len(position_tasks):]):
            valuations[k].update(rows)

        per_scenario = (time.time() - batch_start) / len(scenarios)
        results = []
        for scenario, scenario_envs, rows in zip(scenarios, envs, valuations):
            ordered = {position_id: rows[position_id] for position_id in portfolio.positions}
            result = self._assemble_result(
                portfolio, self._view(portfolio, scenario_envs), scenario, ordered, originals, baseline_value
            )
            result.execution_time = per_scenario
            results.append(result)
        return results, execution, workers

    @property
    def _valuation_mode(self) -> str:
        return "native_batch" if self.config.native_batch else "scalar"

    @property
    def _use_analytical(self) -> bool:
        return self.config.greeks_method == "analytical"

    def _value_positions(
        self,
        positions: Dict[str, Any],
        envs: Dict[str, Any],
    ) -> Dict[str, PositionValuation]:
        """Each position valued once under its underlying's environment."""
        return value_positions(
            positions, envs, self.greeks_calculator, use_analytical=self._use_analytical
        )

    def _evaluate_scalar(
        self,
        portfolio: Portfolio,
        scenario: Scenario,
        originals: Dict[str, float],
        baseline_value: float,
    ) -> ScenarioResult:
        """One scenario on a fresh copy of the positions; each (position, environment) priced once (spec D5)."""
        scenario_start = time.time()
        stressed_envs = StressApplicator.apply_scenario_to_portfolio(portfolio, scenario)
        stressed_portfolio = self._create_stressed_portfolio(portfolio, stressed_envs)
        valuations = self._value_positions(stressed_portfolio.positions, stressed_envs)
        result = self._assemble_result(
            portfolio, stressed_portfolio, scenario, valuations, originals, baseline_value
        )
        result.execution_time = time.time() - scenario_start
        return result

    def _assemble_result(
        self,
        portfolio: Portfolio,
        stressed_portfolio: Portfolio,
        scenario: Scenario,
        valuations: Dict[str, PositionValuation],
        originals: Dict[str, float],
        baseline_value: float,
    ) -> ScenarioResult:
        """Portfolio total, position rows and per-underlying totals, all from the same valuations."""
        stressed_value = sum(v.value for v in valuations.values())
        portfolio_pnl = stressed_value - baseline_value
        portfolio_pnl_pct = pnl_pct_of_abs_baseline(portfolio_pnl, baseline_value)

        greeks = None
        if self.greeks_calculator is not None:
            greeks = aggregate_greeks(v.greeks for v in valuations.values())

        position_results: List[Dict[str, Any]] = []
        if self.config.save_detailed_results:
            position_results = self._position_rows(stressed_portfolio, valuations, originals)

        underlying_results = self._underlying_rows(stressed_portfolio, valuations)

        extra_metrics: Dict[str, Dict[str, Any]] = {}
        if self.metrics_adapter and self.metrics_adapter.supports(portfolio):
            extra_metrics = self.metrics_adapter.compute_metrics(
                portfolio,
                stressed_portfolio,
                scenario,
                baseline_value,
                stressed_value,
            )

        return ScenarioResult(
            scenario=scenario,
            portfolio_value=stressed_value,
            portfolio_pnl=portfolio_pnl,
            portfolio_pnl_pct=portfolio_pnl_pct,
            greeks=greeks,
            position_results=position_results,
            underlying_results=underlying_results,
            extra_metrics=extra_metrics,
            metadata={"valuation_mode": self._valuation_mode},
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

    def _position_rows(
        self,
        stressed_portfolio: Portfolio,
        valuations: Dict[str, PositionValuation],
        originals: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        rows = []
        for position_id, position in stressed_portfolio.positions.items():
            original_value = originals[position_id]
            stressed_value = valuations[position_id].value
            position_pnl = stressed_value - original_value
            row = {
                "position_id": position_id,
                "underlying": position.underlying,
                "product_type": position.product.__class__.__name__,
                "quantity": position.quantity,
                "original_value": original_value,
                "stressed_value": stressed_value,
                "pnl": position_pnl,
                "pnl_pct": pnl_pct_of_abs_baseline(position_pnl, original_value),
            }
            if self.greeks_calculator is not None:
                row["greeks"] = valuations[position_id].greeks
            rows.append(row)
        return rows

    def _underlying_rows(
        self,
        stressed_portfolio: Portfolio,
        valuations: Dict[str, PositionValuation],
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for underlying in stressed_portfolio.pricing_environments.keys():
            ids = [
                position_id
                for position_id, position in stressed_portfolio.positions.items()
                if position.underlying == underlying
            ]
            if not ids:
                continue
            greeks = None
            if self.greeks_calculator is not None:
                greeks = aggregate_greeks(valuations[i].greeks for i in ids)
            results[underlying] = {
                "num_positions": len(ids),
                "total_value": sum(valuations[i].value for i in ids),
                "greeks": greeks,
            }
        return results

    def __repr__(self) -> str:
        return f"EquityStressEngine(config={self.config})"


# Backward-compatible alias to avoid breaking external imports.
StressTestEngine = EquityStressEngine
