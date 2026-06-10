"""
Fixed Income backtest configuration class.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from quantark.util.exceptions import ValidationError


@dataclass
class FIBacktestConfig:
    """
    Configuration for Fixed Income backtest execution.
    
    This class bundles all parameters needed to run an FI backtest, including
    strategy, date range, market data source, cost models, and hedge instrument.
    
    Attributes:
        strategy: Strategy instance implementing BaseStrategy (DV01Neutral, etc.)
        start_date: Start date for backtest
        end_date: End date for backtest
        underlying: Underlying identifier (e.g., 'UST_10Y')
        initial_positions: List of initial FI positions
        market_data_adapter: Adapter for fetching FI market data
        transaction_cost_model: Model for transaction costs
        hedge_futures_spec: Specification for hedge bond futures
        frequency: Data frequency ('D' for daily)
        currency: Currency for rates (default: 'USD')
        logging_level: Logging verbosity
        results_path: Optional path to save results
        calculate_risk_measures: Whether to calculate DV01/convexity at each step
        metadata: Additional metadata for the backtest
    """
    strategy: Any  # BaseStrategy
    start_date: datetime
    end_date: datetime
    underlying: str
    initial_positions: List[Any]  # List[FIPosition]
    market_data_adapter: Any  # BaseMarketDataAdapter
    transaction_cost_model: Any  # TransactionCostModel
    hedge_futures_spec: Optional[Any] = None  # BondFutures specification
    frequency: str = 'D'
    currency: str = 'USD'
    logging_level: str = 'INFO'
    results_path: Optional[str] = None
    save_snapshots: bool = True
    calculate_risk_measures: bool = True
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
        
        if self.frequency not in ['D', 'H', 'M', 'W']:
            raise ValidationError(
                f"Invalid frequency '{self.frequency}'. Must be 'D', 'H', 'M', or 'W'"
            )
        
        if self.logging_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValidationError(
                f"Invalid logging level '{self.logging_level}'"
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
            'strategy': self.strategy.__class__.__name__ if self.strategy else None,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'underlying': self.underlying,
            'num_initial_positions': len(self.initial_positions),
            'frequency': self.frequency,
            'transaction_cost_model': self.transaction_cost_model.__class__.__name__ if self.transaction_cost_model else None,
            'calculate_risk_measures': self.calculate_risk_measures,
            'hedge_futures': repr(self.hedge_futures_spec) if self.hedge_futures_spec else None,
            'asset_class': 'fixed_income',
            'metadata': self.metadata
        }
    
    def validate(self) -> None:
        """Validate the configuration."""
        self._validate()
    
    def __repr__(self) -> str:
        return (
            f"FIBacktestConfig("
            f"strategy={self.strategy.__class__.__name__ if self.strategy else None}, "
            f"period={self.start_date.date()} to {self.end_date.date()}, "
            f"underlying={self.underlying})"
        )

