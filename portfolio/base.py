"""
Base protocols for portfolio and position management.

These protocols define the interface that all asset-class-specific implementations
must follow, enabling a unified API across equity, fixed income, and other asset classes.
"""
from typing import Protocol, Dict, Any, Optional, List, runtime_checkable
from datetime import datetime
from abc import abstractmethod


@runtime_checkable
class BasePosition(Protocol):
    """
    Protocol defining the interface for all position types.
    
    A position represents a holding in a specific instrument with quantity,
    entry details, and the ability to calculate current value and risk measures.
    """
    
    position_id: str
    quantity: float
    entry_price: float
    underlying: str
    entry_timestamp: datetime
    
    def get_current_price(self, pricing_context: Any) -> float:
        """
        Get current market price of the instrument.
        
        Args:
            pricing_context: Asset-specific pricing context (PricingEnvironment, etc.)
            
        Returns:
            Current market price
        """
        ...
    
    def get_market_value(self, pricing_context: Any) -> float:
        """
        Calculate current market value of the position.
        
        Args:
            pricing_context: Asset-specific pricing context
            
        Returns:
            Current market value (price × quantity)
        """
        ...
    
    def get_pnl(self, pricing_context: Any) -> float:
        """
        Calculate unrealized P&L of the position.
        
        Args:
            pricing_context: Asset-specific pricing context
            
        Returns:
            Unrealized profit/loss
        """
        ...
    
    def get_risk_measures(self, pricing_context: Any, **kwargs) -> Dict[str, float]:
        """
        Calculate risk measures for the position.
        
        For equity: delta, gamma, vega, theta, rho
        For fixed income: dv01, convexity, modified_duration
        
        Args:
            pricing_context: Asset-specific pricing context
            **kwargs: Additional parameters for risk calculation
            
        Returns:
            Dictionary of risk measures
        """
        ...
    
    def is_long(self) -> bool:
        """Check if position is long (positive quantity)."""
        ...
    
    def is_short(self) -> bool:
        """Check if position is short (negative quantity)."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize position to dictionary.
        
        Returns:
            Dictionary representation of position
        """
        ...


@runtime_checkable
class BasePortfolio(Protocol):
    """
    Protocol defining the interface for all portfolio types.
    
    A portfolio manages a collection of positions and provides methods for
    aggregating values and risk measures across all positions.
    """
    
    portfolio_name: str
    positions: Dict[str, Any]  # position_id -> Position
    
    def add_position(self, **kwargs) -> Any:
        """
        Add a new position to the portfolio.
        
        Args:
            **kwargs: Asset-specific position parameters
            
        Returns:
            The created position
        """
        ...
    
    def remove_position(self, position_id: str) -> Optional[Any]:
        """
        Remove a position from the portfolio.
        
        Args:
            position_id: ID of position to remove
            
        Returns:
            Removed position or None if not found
        """
        ...
    
    def update_position(self, position_id: str, **kwargs) -> Any:
        """
        Update an existing position.
        
        Args:
            position_id: ID of position to update
            **kwargs: Fields to update
            
        Returns:
            Updated position
        """
        ...
    
    def get_position(self, position_id: str) -> Optional[Any]:
        """
        Get a position by ID.
        
        Args:
            position_id: Position identifier
            
        Returns:
            Position or None if not found
        """
        ...
    
    def get_portfolio_value(self, as_of_date: Optional[datetime] = None) -> float:
        """
        Calculate total portfolio market value.
        
        Args:
            as_of_date: Valuation date (optional)
            
        Returns:
            Total portfolio value
        """
        ...
    
    def get_portfolio_pnl(self) -> float:
        """
        Calculate total portfolio unrealized P&L.
        
        Returns:
            Total unrealized P&L
        """
        ...
    
    def get_portfolio_risk_measures(self, **kwargs) -> Dict[str, float]:
        """
        Calculate aggregated risk measures across all positions.
        
        Args:
            **kwargs: Asset-specific parameters
            
        Returns:
            Dictionary of aggregated risk measures
        """
        ...
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary statistics.
        
        Returns:
            Dictionary with summary information
        """
        ...
    
    def __len__(self) -> int:
        """Return number of positions in portfolio."""
        ...

