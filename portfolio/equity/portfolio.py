"""
Equity portfolio class for managing multiple equity derivative positions.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from .position import EquityPosition
from priceenv import PricingEnvironment
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.engine.base_engine import BaseEngine
from asset.equity.riskmeasures import GreeksCalculator
from util.exceptions import ValidationError


@dataclass
class EquityPortfolio:
    """
    Portfolio container for managing multiple equity derivative positions.
    
    A portfolio tracks positions across multiple underlyings, each with its
    own pricing environment. It provides position management, valuation,
    and risk aggregation capabilities.
    
    Attributes:
        portfolio_name: Name identifier for the portfolio
        positions: Dictionary of positions keyed by position_id
        pricing_environments: Dictionary of pricing environments keyed by underlying
        creation_date: When the portfolio was created
    """
    portfolio_name: str
    pricing_environments: Dict[str, PricingEnvironment]
    creation_date: datetime = field(default_factory=datetime.now)
    positions: Dict[str, EquityPosition] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate portfolio parameters."""
        if not self.portfolio_name:
            raise ValidationError("Portfolio name is required")
        if not isinstance(self.pricing_environments, dict):
            raise ValidationError("pricing_environments must be a dictionary")
    
    def add_position(
        self,
        product: BaseEquityProduct,
        quantity: float,
        entry_price: float,
        underlying: str,
        engine: BaseEngine,
        entry_timestamp: Optional[datetime] = None
    ) -> EquityPosition:
        """
        Add a new position to the portfolio.
        
        Args:
            product: The derivative product
            quantity: Number of contracts (positive=long, negative=short)
            entry_price: Entry price for the position
            underlying: Underlying asset identifier
            engine: Pricing engine for this position
            entry_timestamp: Entry time (defaults to now)
            
        Returns:
            The created Position object
            
        Raises:
            ValidationError: If underlying not in pricing_environments
        """
        if underlying not in self.pricing_environments:
            raise ValidationError(
                f"Underlying '{underlying}' not found in pricing environments. "
                f"Available: {list(self.pricing_environments.keys())}"
            )
        
        if entry_timestamp is None:
            entry_timestamp = datetime.now()
        
        position = EquityPosition(
            product=product,
            quantity=quantity,
            entry_price=entry_price,
            underlying=underlying,
            engine=engine,
            entry_timestamp=entry_timestamp
        )
        
        self.positions[position.position_id] = position
        return position
    
    def remove_position(self, position_id: str) -> Optional[EquityPosition]:
        """
        Remove a position from the portfolio.
        
        Args:
            position_id: ID of the position to remove
            
        Returns:
            The removed Position object, or None if not found
        """
        return self.positions.pop(position_id, None)
    
    def update_position(
        self,
        position_id: str,
        quantity: Optional[float] = None,
        entry_price: Optional[float] = None
    ) -> EquityPosition:
        """
        Update an existing position.
        
        Args:
            position_id: ID of the position to update
            quantity: New quantity (optional)
            entry_price: New entry price (optional)
            
        Returns:
            The updated Position object
            
        Raises:
            KeyError: If position_id not found
            ValidationError: If updated values are invalid
        """
        if position_id not in self.positions:
            raise KeyError(f"Position {position_id} not found")
        
        position = self.positions[position_id]
        
        if quantity is not None:
            if quantity == 0:
                raise ValidationError("Quantity cannot be zero. Use remove_position() instead.")
            position.quantity = quantity
        
        if entry_price is not None:
            if entry_price < 0:
                raise ValidationError(f"Entry price must be non-negative, got {entry_price}")
            position.entry_price = entry_price
        
        return position
    
    def get_position(self, position_id: str) -> Optional[EquityPosition]:
        """
        Get a position by ID.
        
        Args:
            position_id: Position identifier
            
        Returns:
            Position object or None if not found
        """
        return self.positions.get(position_id)
    
    def get_positions_by_underlying(self, underlying: str) -> List[EquityPosition]:
        """
        Get all positions for a specific underlying.
        
        Args:
            underlying: Underlying asset identifier
            
        Returns:
            List of positions for the underlying
        """
        return [
            pos for pos in self.positions.values()
            if pos.underlying == underlying
        ]
    
    def get_portfolio_value(self, as_of_date: Optional[datetime] = None) -> float:
        """
        Calculate total portfolio market value.
        
        Sums market values across all positions using their respective
        pricing environments and engines.
        
        Args:
            as_of_date: Valuation date (currently not used, for future enhancement)
            
        Returns:
            Total portfolio market value
        """
        total_value = 0.0
        
        for position in self.positions.values():
            pricing_env = self.pricing_environments[position.underlying]
            total_value += position.get_market_value(pricing_env)
        
        return total_value
    
    def get_portfolio_pnl(self) -> float:
        """
        Calculate total portfolio unrealized P&L.
        
        Sums P&L across all positions.
        
        Returns:
            Total unrealized profit/loss
        """
        total_pnl = 0.0
        
        for position in self.positions.values():
            pricing_env = self.pricing_environments[position.underlying]
            total_pnl += position.get_pnl(pricing_env)
        
        return total_pnl
    
    def get_portfolio_greeks(
        self,
        greeks_calculator: GreeksCalculator,
        use_analytical: bool = True
    ) -> Dict[str, float]:
        """
        Calculate aggregated Greeks across all positions.
        
        Each position uses its own engine for pricing and Greeks calculation.
        
        Args:
            greeks_calculator: Greeks calculator instance
            use_analytical: Whether to use analytical Greeks when possible
            
        Returns:
            Dictionary of aggregated Greeks
        """
        aggregated = {
            'market_value': 0.0,
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'rho': 0.0
        }
        
        for position in self.positions.values():
            pricing_env = self.pricing_environments[position.underlying]
            position_greeks = position.get_greeks(
                pricing_env,
                greeks_calculator,
                use_analytical
            )
            
            for key in aggregated:
                if key in position_greeks:
                    aggregated[key] += position_greeks[key]
        
        return aggregated
    
    def get_portfolio_risk_measures(
        self,
        greeks_calculator: Optional[GreeksCalculator] = None,
        use_analytical: bool = True
    ) -> Dict[str, float]:
        """
        Calculate aggregated risk measures across all positions.
        
        For equity portfolios, this returns Greeks.
        
        Args:
            greeks_calculator: Greeks calculator instance
            use_analytical: Whether to use analytical Greeks when possible
            
        Returns:
            Dictionary of aggregated risk measures
        """
        if greeks_calculator is None:
            greeks_calculator = GreeksCalculator()
        return self.get_portfolio_greeks(greeks_calculator, use_analytical)
    
    def get_greeks_by_underlying(
        self,
        underlying: str,
        greeks_calculator: GreeksCalculator,
        use_analytical: bool = True
    ) -> Dict[str, float]:
        """
        Calculate aggregated Greeks for a specific underlying.
        
        Args:
            underlying: Underlying asset identifier
            greeks_calculator: Greeks calculator instance
            use_analytical: Whether to use analytical Greeks when possible
            
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
        pricing_env = self.pricing_environments.get(underlying)
        
        if pricing_env is None:
            return aggregated
        
        for position in positions:
            position_greeks = position.get_greeks(
                pricing_env,
                greeks_calculator,
                use_analytical
            )
            
            for key in aggregated:
                if key in position_greeks:
                    aggregated[key] += position_greeks[key]
        
        return aggregated
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert portfolio positions to pandas DataFrame.
        
        Returns:
            DataFrame with position details and current values
        """
        if not self.positions:
            return pd.DataFrame()
        
        rows = []
        for position in self.positions.values():
            pricing_env = self.pricing_environments[position.underlying]
            
            row = position.to_dict()
            row['current_price'] = position.get_current_price(pricing_env)
            row['market_value'] = position.get_market_value(pricing_env)
            row['pnl'] = position.get_pnl(pricing_env)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary statistics.
        
        Returns:
            Dictionary with summary information
        """
        if not self.positions:
            return {
                'portfolio_name': self.portfolio_name,
                'creation_date': self.creation_date.isoformat(),
                'num_positions': 0,
                'num_underlyings': len(self.pricing_environments),
                'total_value': 0.0,
                'total_pnl': 0.0,
                'long_positions': 0,
                'short_positions': 0,
                'asset_class': 'equity',
            }
        
        total_value = self.get_portfolio_value()
        total_pnl = self.get_portfolio_pnl()
        
        long_positions = sum(1 for pos in self.positions.values() if pos.is_long())
        short_positions = sum(1 for pos in self.positions.values() if pos.is_short())
        
        # Group by underlying
        underlyings_used = set(pos.underlying for pos in self.positions.values())
        
        return {
            'portfolio_name': self.portfolio_name,
            'creation_date': self.creation_date.isoformat(),
            'num_positions': len(self.positions),
            'num_underlyings': len(underlyings_used),
            'underlyings': sorted(underlyings_used),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'pnl_percentage': (total_pnl / (total_value - total_pnl) * 100) if (total_value - total_pnl) != 0 else 0.0,
            'asset_class': 'equity',
        }
    
    def __len__(self):
        """Return number of positions in portfolio."""
        return len(self.positions)
    
    def __repr__(self):
        return (
            f"EquityPortfolio(name={self.portfolio_name}, "
            f"positions={len(self.positions)}, "
            f"underlyings={len(self.pricing_environments)})"
        )


# Backward compatibility alias
Portfolio = EquityPortfolio

