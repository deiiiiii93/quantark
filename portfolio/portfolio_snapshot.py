"""
Portfolio snapshot for recording portfolio state at a specific point in time.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from .equity.portfolio import Portfolio
from asset.equity.riskmeasures import GreeksCalculator
from util.exceptions import ValidationError


@dataclass
class PortfolioSnapshot:
    """
    Snapshot of portfolio state at a specific point in time.
    
    This class captures the complete state of a portfolio including
    positions, valuations, and risk metrics at a given timestamp.
    Useful for historical tracking and performance analysis.
    
    Attributes:
        timestamp: When the snapshot was taken
        portfolio_name: Name of the portfolio
        positions_data: List of serialized position dictionaries
        total_value: Total portfolio market value at snapshot time
        total_pnl: Total unrealized P&L at snapshot time
        aggregated_greeks: Portfolio-level Greeks
        metadata: Additional metadata (e.g., market conditions)
    """
    timestamp: datetime
    portfolio_name: str
    positions_data: List[Dict[str, Any]]
    total_value: float
    total_pnl: float
    aggregated_greeks: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate snapshot parameters."""
        if not self.portfolio_name:
            raise ValidationError("Portfolio name is required")
        if not isinstance(self.positions_data, list):
            raise ValidationError("positions_data must be a list")
        if not isinstance(self.aggregated_greeks, dict):
            raise ValidationError("aggregated_greeks must be a dictionary")
    
    @classmethod
    def from_portfolio(
        cls,
        portfolio: Portfolio,
        greeks_calculator: GreeksCalculator,
        use_analytical: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'PortfolioSnapshot':
        """
        Create a snapshot from a live portfolio.
        
        Args:
            portfolio: Portfolio to snapshot
            greeks_calculator: Greeks calculator for risk metrics
            use_analytical: Whether to use analytical Greeks when possible
            metadata: Additional metadata to store
            
        Returns:
            PortfolioSnapshot instance
        """
        timestamp = datetime.now()
        
        # Serialize positions with current values
        positions_data = []
        for position in portfolio.positions.values():
            pricing_env = portfolio.pricing_environments[position.underlying]
            pos_dict = position.to_dict()
            
            # Add current market data
            pos_dict['snapshot_timestamp'] = timestamp.isoformat()
            pos_dict['current_price'] = position.get_current_price(pricing_env)
            pos_dict['market_value'] = position.get_market_value(pricing_env)
            pos_dict['pnl'] = position.get_pnl(pricing_env)
            
            # Add position-level Greeks
            try:
                pos_greeks = position.get_greeks(
                    pricing_env,
                    greeks_calculator,
                    use_analytical
                )
                pos_dict['greeks'] = pos_greeks
            except Exception as e:
                pos_dict['greeks'] = {}
                pos_dict['greeks_error'] = str(e)
            
            positions_data.append(pos_dict)
        
        # Calculate portfolio-level metrics
        total_value = portfolio.get_portfolio_value()
        total_pnl = portfolio.get_portfolio_pnl()
        
        # Calculate aggregated Greeks
        try:
            aggregated_greeks = portfolio.get_portfolio_greeks(
                greeks_calculator,
                use_analytical
            )
        except Exception as e:
            aggregated_greeks = {
                'error': str(e),
                'market_value': total_value
            }
        
        # Add portfolio summary to metadata
        if metadata is None:
            metadata = {}
        
        metadata['num_positions'] = len(portfolio.positions)
        metadata['num_underlyings'] = len(set(
            pos.underlying for pos in portfolio.positions.values()
        ))
        metadata['creation_date'] = portfolio.creation_date.isoformat()
        
        return cls(
            timestamp=timestamp,
            portfolio_name=portfolio.portfolio_name,
            positions_data=positions_data,
            total_value=total_value,
            total_pnl=total_pnl,
            aggregated_greeks=aggregated_greeks,
            metadata=metadata
        )
    
    def get_positions_by_underlying(self, underlying: str) -> List[Dict[str, Any]]:
        """
        Get positions for a specific underlying from snapshot.
        
        Args:
            underlying: Underlying asset identifier
            
        Returns:
            List of position dictionaries for the underlying
        """
        return [
            pos for pos in self.positions_data
            if pos.get('underlying') == underlying
        ]
    
    def get_greeks_by_underlying(self, underlying: str) -> Dict[str, float]:
        """
        Calculate aggregated Greeks for a specific underlying from snapshot.
        
        Args:
            underlying: Underlying asset identifier
            
        Returns:
            Dictionary of aggregated Greeks for the underlying
        """
        aggregated = {
            'market_value': 0.0,
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'rho': 0.0
        }
        
        positions = self.get_positions_by_underlying(underlying)
        
        for pos in positions:
            greeks = pos.get('greeks', {})
            for key in aggregated:
                if key in greeks:
                    aggregated[key] += greeks[key]
        
        return aggregated
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize snapshot to dictionary.
        
        Returns:
            Dictionary representation of snapshot
        """
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_name': self.portfolio_name,
            'total_value': self.total_value,
            'total_pnl': self.total_pnl,
            'aggregated_greeks': self.aggregated_greeks,
            'num_positions': len(self.positions_data),
            'positions': self.positions_data,
            'metadata': self.metadata
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of the snapshot without position details.
        
        Returns:
            Summary dictionary
        """
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_name': self.portfolio_name,
            'total_value': self.total_value,
            'total_pnl': self.total_pnl,
            'pnl_percentage': (self.total_pnl / (self.total_value - self.total_pnl) * 100) 
                             if (self.total_value - self.total_pnl) != 0 else 0.0,
            'num_positions': len(self.positions_data),
            'aggregated_greeks': self.aggregated_greeks,
            'metadata': self.metadata
        }
    
    def __repr__(self):
        return (
            f"PortfolioSnapshot("
            f"portfolio={self.portfolio_name}, "
            f"timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"value=${self.total_value:.2f}, "
            f"pnl=${self.total_pnl:.2f}, "
            f"positions={len(self.positions_data)})"
        )

