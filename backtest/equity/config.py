"""
Backtest configuration class.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from util.exceptions import ValidationError


@dataclass
class BacktestConfig:
    """
    Configuration for backtest execution.

    This class bundles all parameters needed to run a backtest, including
    strategy, date range, market data source, and cost models.

    Attributes:
        strategy: Strategy instance implementing BaseStrategy
        start_date: Start date for backtest
        end_date: End date for backtest
        underlying: Underlying asset identifier
        initial_positions: List of initial positions to start with
        market_data_adapter: Adapter for fetching market data
        transaction_cost_model: Model for transaction costs
        frequency: Data frequency ('D' for daily, 'H' for hourly)
        currency: Currency for rates (default: 'USD')
        logging_level: Logging verbosity ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        results_path: Optional path to save results
        save_snapshots: Whether to save portfolio snapshots at each step
        calculate_greeks: Whether to calculate Greeks at each step
        greeks_method: Method for Greeks calculation ('analytical' or 'numerical')
        metadata: Additional metadata for the backtest
    """

    strategy: Any  # Will be BaseStrategy, but avoid circular import
    start_date: datetime
    end_date: datetime
    underlying: str
    initial_positions: List[Any]  # List[Position]
    market_data_adapter: Any  # BaseMarketDataAdapter
    transaction_cost_model: Any  # TransactionCostModel
    frequency: str = "D"
    currency: str = "USD"
    logging_level: str = "INFO"
    results_path: Optional[str] = None
    save_snapshots: bool = True
    calculate_greeks: bool = True
    greeks_method: str = "analytical"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate()

    def _validate(self):
        """
        Validate configuration parameters.

        Raises:
            ValidationError: If any parameters are invalid
        """
        if self.start_date >= self.end_date:
            raise ValidationError(
                f"Start date {self.start_date} must be before end date {self.end_date}"
            )

        if not self.underlying:
            raise ValidationError("Underlying identifier is required")

        if self.frequency not in ["D", "H", "M", "W"]:
            raise ValidationError(
                f"Invalid frequency '{self.frequency}'. Must be 'D', 'H', 'M', or 'W'"
            )

        if self.logging_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValidationError(f"Invalid logging level '{self.logging_level}'")

        if self.greeks_method not in ["analytical", "numerical"]:
            raise ValidationError(
                f"Invalid greeks_method '{self.greeks_method}'. Must be 'analytical' or 'numerical'"
            )

        if self.initial_positions is None:
            self.initial_positions = []

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of configuration.

        Returns:
            Dictionary with configuration summary
        """
        return {
            "strategy": self.strategy.__class__.__name__ if self.strategy else None,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "underlying": self.underlying,
            "num_initial_positions": len(self.initial_positions),
            "frequency": self.frequency,
            "transaction_cost_model": self.transaction_cost_model.__class__.__name__,
            "calculate_greeks": self.calculate_greeks,
            "greeks_method": self.greeks_method,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"BacktestConfig("
            f"strategy={self.strategy.__class__.__name__ if self.strategy else None}, "
            f"period={self.start_date.date()} to {self.end_date.date()}, "
            f"underlying={self.underlying})"
        )


# Alias for explicit naming
EquityBacktestConfig = BacktestConfig
