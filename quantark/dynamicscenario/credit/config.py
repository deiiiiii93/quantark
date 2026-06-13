"""Configuration for credit dynamic scenario analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from quantark.util.exceptions import ValidationError


@dataclass
class CreditDynamicScenarioConfig:
    """Configuration for credit multi-day scenario analysis."""

    calculate_greeks: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["parquet"])
    save_detailed_results: bool = True
    generate_report: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = {"parquet", "csv", "json", "html"}
        for fmt in self.export_formats:
            if fmt not in valid:
                raise ValidationError(
                    f"Invalid export format '{fmt}'. Valid: {sorted(valid)}"
                )

    def get_summary(self) -> Dict[str, Any]:
        return {
            "asset_class": "credit",
            "calculate_greeks": self.calculate_greeks,
            "export_formats": self.export_formats,
            "save_detailed_results": self.save_detailed_results,
            "generate_report": self.generate_report,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"CreditDynamicScenarioConfig(calculate_greeks={self.calculate_greeks})"
