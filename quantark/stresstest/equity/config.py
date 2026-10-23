"""Equity stress configuration implementation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from quantark.util.exceptions import ValidationError


@dataclass
class EquityStressConfig:
    """Configuration for equity stress test execution."""

    calculate_greeks: bool = True
    greeks_method: str = "analytical"
    export_formats: List[str] = field(default_factory=lambda: ["parquet"])
    output_dir: str = "./stress_results"
    save_detailed_results: bool = True

    # Opt-in native batching: positions whose engine supports a spot-batch session reuse one prepared
    # context across scenarios (results within the route's measured tolerance of the scalar path)
    native_batch: bool = False

    # Static scenarios on a spawn process pool (bitwise equal to serial); a callback stays in the parent.
    # A script that runs with it must guard its entry point (``if __name__ == "__main__":``)
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
            "native_batch": self.native_batch,
            "parallel_execution": self.parallel_execution,
            "max_workers": self.max_workers,
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

