"""
Base classes and protocols for VaR engines.
"""

from typing import Any, Protocol, Union, runtime_checkable

import pandas as pd

from quantark.var.results import IncrementalVaRResult, VaRResult


@runtime_checkable
class VaREngine(Protocol):
    """Protocol defining the interface for VaR calculation engines."""

    def calculate_var(
        self,
        portfolio: Any,
        historical_data: Union[Any, pd.DataFrame],
    ) -> "VaRResult":
        """
        Calculate VaR for the portfolio.

        Args:
            portfolio: Portfolio object (EquityPortfolio or FIPortfolio)
            historical_data: Historical market data (MarketDataSet or DataFrame)

        Returns:
            VaRResult containing VaR metrics and attribution
        """
        ...

    def calculate_incremental_var(
        self,
        portfolio: Any,
        historical_data: Union[Any, pd.DataFrame],
    ) -> "IncrementalVaRResult":
        """
        Calculate Incremental VaR for the portfolio.

        Calculates the contribution of each position to total portfolio VaR
        by measuring the change in VaR when each position is excluded.

        Args:
            portfolio: Portfolio object (EquityPortfolio or FIPortfolio)
            historical_data: Historical market data (MarketDataSet or DataFrame)

        Returns:
            IncrementalVaRResult with position-level IVaR analysis
        """
        ...

    def supports_portfolio(self, portfolio: Any) -> bool:
        """
        Check if this engine supports the portfolio type.

        Args:
            portfolio: Portfolio object to check

        Returns:
            True if supported, False otherwise
        """
        ...
