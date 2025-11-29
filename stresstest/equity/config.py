"""Equity stress configuration implementation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from util.exceptions import ValidationError


@dataclass
class EquityStressConfig:
    """Configuration for equity stress test execution."""

    calculate_greeks: bool = True
    greeks_method: str = "analytical"
    export_formats: List[str] = field(default_factory=lambda: ["parquet"])
    output_dir: str = "./stress_results"
    save_detailed_results: bool = True

    # Future features for dynamic scenario analysis
    parallel_execution: bool = False
    max_workers: int = 4
    progress_callback: Optional[Any] = None

    # Additional settings
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.greeks_method not in ["analytical", "numerical"]:
            raise ValidationError(
                f"Invalid greeks_method '{self.greeks_method}'. Must be 'analytical' or 'numerical'"
            )

        valid_formats = ["parquet", "csv", "json", "html"]
        for fmt in self.export_formats:
            if fmt not in valid_formats:
                raise ValidationError(
                    f"Invalid export format '{fmt}'. Valid formats: {valid_formats}"
                )

        if self.max_workers < 1:
            raise ValidationError("max_workers must be at least 1")

    def get_summary(self) -> Dict[str, Any]:
        return {
            "calculate_greeks": self.calculate_greeks,
            "greeks_method": self.greeks_method,
            "export_formats": self.export_formats,
            "output_dir": self.output_dir,
            "save_detailed_results": self.save_detailed_results,
            "parallel_execution": self.parallel_execution,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            "EquityStressConfig("
            f"greeks={self.calculate_greeks}, "
            f"formats={self.export_formats}, "
            f"output_dir='{self.output_dir}'"
            ")"
        )


# Backward-compatible alias used throughout the existing equity workflow.
StressTestConfig = EquityStressConfig

