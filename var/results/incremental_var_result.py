"""
Incremental VaR result classes.

This module contains classes for storing incremental VaR calculations,
which measure the contribution of individual positions to total portfolio VaR.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from var.results.var_result import VaRResult


@dataclass
class IncrementalVaRResult:
    """
    Results from an incremental VaR calculation.

    This class stores the incremental VaR contribution of each position
    and calculates diversification benefits.

    Attributes:
        portfolio_var: Total portfolio VaR
        position_ivari: Incremental VaR by position ID
        diversification_benefit: Reduction in VaR due to diversification
        calculation_timestamp: When calculation was performed
        portfolio_var_without_position: VaR when each position is excluded
        ivari_method: Method used for incremental VaR calculation
        config: Configuration parameters used
    """

    portfolio_var: float
    position_ivari: Dict[str, float]
    diversification_benefit: float

    calculation_timestamp: datetime = field(default_factory=datetime.now)

    portfolio_var_without_position: Optional[Dict[str, float]] = None
    ivari_method: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """
        Post-initialization validation and calculations.
        """
        # Ensure VaR is non-negative
        if self.portfolio_var < 0:
            raise ValueError(f"Portfolio VaR must be non-negative, got {self.portfolio_var}")

        # Calculate diversification benefit if not provided
        if self.diversification_benefit is None:
            self._calculate_diversification_benefit()

        # Ensure all incremental VaR values are valid
        for pos_id, ivari in self.position_ivari.items():
            if ivari < 0:
                raise ValueError(f"Incremental VaR for {pos_id} must be non-negative, got {ivari}")

    def _calculate_diversification_benefit(self) -> None:
        """
        Calculate diversification benefit.

        Diversification benefit = Sum of individual position VaRs - Portfolio VaR
        """
        total_individual_var = sum(self.position_ivari.values())
        self.diversification_benefit = total_individual_var - self.portfolio_var

    def get_position_ivari_as_pct(self, position_value: float) -> float:
        """
        Convert incremental VaR to percentage of position value.

        Args:
            position_value: Market value of the position

        Returns:
            Incremental VaR as percentage
        """
        if position_value <= 0:
            return 0.0
        return self.position_ivari.get(position_value, 0.0) / abs(position_value)

    def get_diversification_ratio(self) -> float:
        """
        Calculate diversification ratio.

        Diversification Ratio = Portfolio VaR / Sum of Individual VaRs

        Returns:
            Ratio between 0 and 1 (1 = no diversification benefit)
        """
        total_individual_var = sum(self.position_ivari.values())
        if total_individual_var == 0:
            return 1.0
        return self.portfolio_var / total_individual_var

    def get_top_contributors(self, n: int = 5) -> List[tuple[str, float]]:
        """
        Get positions with highest incremental VaR contribution.

        Args:
            n: Number of top contributors to return

        Returns:
            List of (position_id, incremental_var) tuples sorted by contribution
        """
        sorted_positions = sorted(
            self.position_ivari.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_positions[:n]

    def get_summary_dict(self) -> Dict[str, Any]:
        """
        Get a summary dictionary of incremental VaR metrics.

        Returns:
            Dictionary with key incremental VaR metrics
        """
        top_contributors = self.get_top_contributors(5)

        return {
            "portfolio_var": self.portfolio_var,
            "diversification_benefit": self.diversification_benefit,
            "diversification_ratio": self.get_diversification_ratio(),
            "num_positions": len(self.position_ivari),
            "top_contributors": top_contributors,
            "ivari_method": self.ivari_method,
            "calculation_timestamp": self.calculation_timestamp.isoformat(),
        }
