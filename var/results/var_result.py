"""
VaR calculation result classes.

This module contains the VaRResult dataclass and related result classes
for storing and reporting VaR calculation outputs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from var.config import VaRMethod


@dataclass
class VaRResult:
    """
    Results from a VaR calculation.

    This class stores all the output from a VaR calculation including
    the VaR value, CVaR, attribution breakdown, scenarios, and metadata.

    Attributes:
        var: Value-at-Risk (loss threshold at confidence level)
        cvar: Conditional Value-at-Risk (expected loss given VaR exceeded)
        confidence_level: Confidence level (e.g., 0.99 for 99%)
        holding_period: Holding period in days
        method: VaR calculation method used
        portfolio_value: Current portfolio market value
        var_as_pct: VaR as percentage of portfolio value
        component_var: Component VaR breakdown by position (Euler allocation)
        marginal_var: Marginal VaR contribution by position
        factor_var: VaR attribution by risk factor
        incremental_var: Incremental VaR by position
        scenarios: DataFrame of all scenarios with P&L
        worst_scenarios: List of worst N scenarios with details
        stressed_var: Stressed VaR (SVaR) value
        stressed_cvar: Stressed CVaR value
        stressed_period: Dict with start/end dates of stressed period
        calculation_timestamp: When calculation was performed
        execution_time_seconds: Time taken to calculate
        config_summary: Summary of configuration parameters
    """

    var: float
    cvar: float
    confidence_level: float
    holding_period: int
    method: VaRMethod

    portfolio_value: float
    var_as_pct: float

    component_var: Optional[Dict[str, float]] = None
    marginal_var: Optional[Dict[str, float]] = None
    factor_var: Optional[Dict[str, float]] = None
    incremental_var: Optional[Dict[str, float]] = None

    scenarios: Optional[pd.DataFrame] = None
    worst_scenarios: Optional[List[Dict]] = None

    stressed_var: Optional[float] = None
    stressed_cvar: Optional[float] = None
    stressed_period: Optional[Dict[str, datetime]] = None

    calculation_timestamp: datetime = field(default_factory=datetime.now)
    execution_time_seconds: float = 0.0
    config_summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Post-initialization validation and calculations.
        """
        # Ensure VaR is non-negative (represents loss)
        if self.var < 0:
            raise ValueError(f"VaR must be non-negative, got {self.var}")

        # Ensure CVaR >= VaR (expected shortfall should be worse)
        if self.cvar < self.var:
            raise ValueError(f"CVaR ({self.cvar}) must be >= VaR ({self.var})")

        # Ensure confidence level is valid
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {self.confidence_level}"
            )

        # Ensure holding period is positive
        if self.holding_period < 1:
            raise ValueError(
                f"Holding period must be >= 1, got {self.holding_period}"
            )

    def get_var_as_currency(self, currency: str = "$") -> str:
        """
        Format VaR as a currency string.

        Args:
            currency: Currency symbol (e.g., "$", "€", "£")

        Returns:
            Formatted VaR string
        """
        return f"{currency}{self.var:.2f}"

    def get_var_as_percentage(self) -> str:
        """
        Format VaR as percentage.

        Returns:
            Formatted VaR percentage string
        """
        return f"{self.var_as_pct * 100:.2f}%"

    def get_summary_dict(self) -> Dict[str, Any]:
        """
        Get a summary dictionary of key metrics.

        Returns:
            Dictionary with key VaR metrics
        """
        return {
            "var": self.var,
            "cvar": self.cvar,
            "confidence_level": self.confidence_level,
            "holding_period": self.holding_period,
            "method": str(self.method),
            "portfolio_value": self.portfolio_value,
            "var_as_pct": self.var_as_pct,
            "stressed_var": self.stressed_var,
            "execution_time_seconds": self.execution_time_seconds,
        }
