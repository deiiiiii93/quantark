"""Equity stress result types."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from quantark.stresstest.base import ScenarioEnvelope, StressResultEnvelope


@dataclass
class ScenarioResult(ScenarioEnvelope):
    """Scenario-level result with helper utilities."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario.name,
            "scenario_description": self.scenario.description,
            "portfolio_value": self.portfolio_value,
            "portfolio_pnl": self.portfolio_pnl,
            "portfolio_pnl_pct": self.portfolio_pnl_pct,
            "greeks": self.greeks,
            "num_positions": len(self.position_results),
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "extra_metrics": self.extra_metrics,
        }

    def get_summary(self) -> str:
        lines = [
            f"Scenario: {self.scenario.name}",
            f"Portfolio Value: ${self.portfolio_value:,.2f}",
            f"P&L: ${self.portfolio_pnl:,.2f} ({self.portfolio_pnl_pct:+.2f}%)",
        ]
        if self.greeks:
            lines.append("Greeks:")
            for key, value in self.greeks.items():
                if key != "market_value":
                    lines.append(f"  {key}: {value:,.4f}")
        return "\n".join(lines)


@dataclass
class StressTestResults(StressResultEnvelope):
    """Full equity stress result set."""

    scenario_results: List[ScenarioResult] = field(default_factory=list)

    def get_scenario_result(self, scenario_name: str) -> Optional[ScenarioResult]:
        for result in self.scenario_results:
            if result.scenario.name == scenario_name:
                return result
        return None

    def get_worst_scenario(self) -> Optional[ScenarioResult]:
        if not self.scenario_results:
            return None
        return min(self.scenario_results, key=lambda r: r.portfolio_pnl)

    def get_best_scenario(self) -> Optional[ScenarioResult]:
        if not self.scenario_results:
            return None
        return max(self.scenario_results, key=lambda r: r.portfolio_pnl)

    def to_summary_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        baseline_row: Dict[str, Any] = {
            "scenario_name": "Baseline",
            "scenario_description": "Unstressed (current market)",
            "portfolio_value": self.baseline_value,
            "portfolio_pnl": 0.0,
            "portfolio_pnl_pct": 0.0,
        }
        if self.baseline_greeks:
            for key, value in self.baseline_greeks.items():
                if key != "market_value":
                    baseline_row[f"greek_{key}"] = value
        rows.append(baseline_row)

        for result in self.scenario_results:
            row: Dict[str, Any] = {
                "scenario_name": result.scenario.name,
                "scenario_description": result.scenario.description or "",
                "portfolio_value": result.portfolio_value,
                "portfolio_pnl": result.portfolio_pnl,
                "portfolio_pnl_pct": result.portfolio_pnl_pct,
            }
            if result.greeks:
                for key, value in result.greeks.items():
                    if key != "market_value":
                        row[f"greek_{key}"] = value
            rows.append(row)

        return pd.DataFrame(rows)

    def to_position_dataframe(self, scenario_name: str) -> Optional[pd.DataFrame]:
        result = self.get_scenario_result(scenario_name)
        if result is None or not result.position_results:
            return None
        return pd.DataFrame(result.position_results)

    def get_summary(self) -> str:
        lines = [
            "=" * 60,
            "STRESS TEST RESULTS SUMMARY",
            "=" * 60,
            f"Execution Time: {self.execution_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Duration: {self.total_execution_time:.2f} seconds",
            f"Number of Scenarios: {len(self.scenario_results)}",
            "",
            f"Baseline Portfolio Value: ${self.baseline_value:,.2f}",
            "",
            "Scenario Results:",
            "-" * 60,
        ]

        for result in self.scenario_results:
            lines.append(f"\n{result.scenario.name}:")
            lines.append(f"  Value: ${result.portfolio_value:,.2f}")
            lines.append(f"  P&L: ${result.portfolio_pnl:,.2f} ({result.portfolio_pnl_pct:+.2f}%)")

        worst = self.get_worst_scenario()
        best = self.get_best_scenario()
        if worst and best:
            lines.extend(
                [
                    "",
                    "-" * 60,
                    f"Worst Scenario: {worst.scenario.name}",
                    f"  P&L: ${worst.portfolio_pnl:,.2f} ({worst.portfolio_pnl_pct:+.2f}%)",
                    "",
                    f"Best Scenario: {best.scenario.name}",
                    f"  P&L: ${best.portfolio_pnl:,.2f} ({best.portfolio_pnl_pct:+.2f}%)",
                ]
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"StressTestResults(scenarios={len(self.scenario_results)}, "
            f"baseline_value=${self.baseline_value:,.2f})"
        )

