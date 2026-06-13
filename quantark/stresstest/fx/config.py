"""Configuration for FX stress testing."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class FXStressConfig:
    """
    Configuration for the FX stress engine.

    Attributes:
        calculate_greeks: Whether to compute portfolio FX greeks per scenario.
        save_detailed_results: Whether to keep per-position result rows.
    """

    calculate_greeks: bool = True
    save_detailed_results: bool = False

    def get_summary(self) -> Dict[str, Any]:
        return {
            "asset_class": "fx",
            "calculate_greeks": self.calculate_greeks,
            "save_detailed_results": self.save_detailed_results,
        }

    def __repr__(self) -> str:
        return (
            f"FXStressConfig(calculate_greeks={self.calculate_greeks}, "
            f"save_detailed_results={self.save_detailed_results})"
        )
