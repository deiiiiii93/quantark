"""FI-specific stress result types."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from quantark.stresstest.equity.results import ScenarioResult, StressTestResults


@dataclass
class FIScenarioResult(ScenarioResult):
    """Scenario result enriched with FI metrics."""

    def get_fi_metrics(self) -> Dict[str, Any]:
        return self.extra_metrics.get("fi", {})


@dataclass
class FIStressResults(StressTestResults):
    """Stress results with FI metrics and helpers."""

    scenario_results: List[FIScenarioResult] = field(default_factory=list)

    def get_dv01_series(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        baseline_metrics = self.extra_metrics.get("fi", {})
        if baseline_metrics:
            rows.append(
                {
                    "scenario": "Baseline",
                    "dv01": baseline_metrics.get("dv01", 0.0),
                    "convexity": baseline_metrics.get("convexity", 0.0),
                    "carry": baseline_metrics.get("carry", 0.0),
                    "modified_duration": baseline_metrics.get("modified_duration", 0.0),
                    "type": "baseline",
                }
            )

        for result in self.scenario_results:
            metrics = result.get_fi_metrics()
            rows.append(
                {
                    "scenario": result.scenario.name,
                    "dv01": metrics.get("dv01", 0.0),
                    "convexity": metrics.get("convexity", 0.0),
                    "carry": metrics.get("carry", 0.0),
                    "modified_duration": metrics.get("modified_duration", 0.0),
                    "type": "stressed",
                }
            )

        return pd.DataFrame(rows)

    def get_curve_shift_summary(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        scenario = self.get_scenario_result(scenario_name)
        if scenario is None:
            return None
        return scenario.get_fi_metrics().get("curve_summary")

    def get_hedge_summary(self) -> Dict[str, Any]:
        return self.extra_metrics.get("hedge", {})

