"""Fixed income stress configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from util.exceptions import ValidationError


@dataclass
class FIStressConfig:
    """Configuration for FI-focused stress tests."""

    dv01_alert_threshold: float = 50_000.0
    convexity_alert_threshold: float = 1_000_000.0
    track_key_rate_dv01: bool = True
    key_rate_buckets: List[str] = field(
        default_factory=lambda: ["2Y", "5Y", "10Y", "30Y"]
    )
    save_dv01_series: bool = True
    include_carry_metrics: bool = True
    save_detailed_results: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["parquet"])
    output_dir: str = "./stress_results"
    hedge_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.track_key_rate_dv01 and not self.key_rate_buckets:
            raise ValidationError("key_rate_buckets must not be empty when tracking key-rate DV01")

        if self.dv01_alert_threshold < 0:
            raise ValidationError("dv01_alert_threshold must be non-negative")

        if self.convexity_alert_threshold < 0:
            raise ValidationError("convexity_alert_threshold must be non-negative")

    def get_summary(self) -> Dict[str, Any]:
        return {
            "dv01_alert_threshold": self.dv01_alert_threshold,
            "convexity_alert_threshold": self.convexity_alert_threshold,
            "track_key_rate_dv01": self.track_key_rate_dv01,
            "key_rate_buckets": self.key_rate_buckets,
            "save_dv01_series": self.save_dv01_series,
            "include_carry_metrics": self.include_carry_metrics,
            "save_detailed_results": self.save_detailed_results,
            "export_formats": self.export_formats,
            "output_dir": self.output_dir,
            "hedge_metadata": self.hedge_metadata,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            "FIStressConfig("
            f"dv01_threshold={self.dv01_alert_threshold}, "
            f"convexity_threshold={self.convexity_alert_threshold}, "
            f"key_rates={self.key_rate_buckets})"
        )

