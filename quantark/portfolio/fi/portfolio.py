"""
Fixed Income portfolio class for managing bond and bond derivative positions.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from .position import FIPosition
from quantark.asset.bond.product.base_bond_product import BaseBondProduct
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError


@dataclass
class FIPortfolio:
    """
    Portfolio container for managing Fixed Income positions.
    
    A portfolio tracks bond positions across multiple issuers/underlyings,
    each with its own pricing environment. Provides FI-specific risk
    aggregation (DV01, duration, convexity).
    
    Attributes:
        portfolio_name: Name identifier for the portfolio
        positions: Dictionary of positions keyed by position_id
        pricing_environments: Dictionary of pricing environments keyed by underlying
        creation_date: When the portfolio was created
    """
    portfolio_name: str
    pricing_environments: Dict[str, PricingEnvironment]
    creation_date: datetime = field(default_factory=datetime.now)
    positions: Dict[str, FIPosition] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate portfolio parameters."""
        if not self.portfolio_name:
            raise ValidationError("Portfolio name is required")
        if not isinstance(self.pricing_environments, dict):
            raise ValidationError("pricing_environments must be a dictionary")
    
    def add_position(
        self,
        product: BaseBondProduct,
        quantity: float,
        entry_price: float,
        underlying: str,
        engine: Any,
        entry_timestamp: Optional[datetime] = None,
        notional_per_unit: float = 100.0
    ) -> FIPosition:
        """
        Add a new Fixed Income position to the portfolio.
        
        Args:
            product: The bond product
            quantity: Number of bonds/contracts (positive=long, negative=short)
            entry_price: Entry clean price for the position
            underlying: Underlying identifier (issuer or bond identifier)
            engine: Pricing engine for this position
            entry_timestamp: Entry time (defaults to now)
            notional_per_unit: Notional per unit (default: 100)
            
        Returns:
            The created FIPosition object
            
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
        
        position = FIPosition(
            product=product,
            quantity=quantity,
            entry_price=entry_price,
            underlying=underlying,
            engine=engine,
            entry_timestamp=entry_timestamp,
            notional_per_unit=notional_per_unit
        )
        
        self.positions[position.position_id] = position
        return position
    
    def remove_position(self, position_id: str) -> Optional[FIPosition]:
        """
        Remove a position from the portfolio.
        
        Args:
            position_id: ID of the position to remove
            
        Returns:
            The removed FIPosition object, or None if not found
        """
        return self.positions.pop(position_id, None)
    
    def update_position(
        self,
        position_id: str,
        quantity: Optional[float] = None,
        entry_price: Optional[float] = None
    ) -> FIPosition:
        """
        Update an existing position.
        
        Args:
            position_id: ID of the position to update
            quantity: New quantity (optional)
            entry_price: New entry price (optional)
            
        Returns:
            The updated FIPosition object
            
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
    
    def get_position(self, position_id: str) -> Optional[FIPosition]:
        """
        Get a position by ID.
        
        Args:
            position_id: Position identifier
            
        Returns:
            FIPosition object or None if not found
        """
        return self.positions.get(position_id)
    
    def get_positions_by_underlying(self, underlying: str) -> List[FIPosition]:
        """
        Get all positions for a specific underlying.
        
        Args:
            underlying: Underlying identifier
            
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
        
        Args:
            as_of_date: Valuation date (currently not used)
            
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
        
        Returns:
            Total unrealized profit/loss
        """
        total_pnl = 0.0
        
        for position in self.positions.values():
            pricing_env = self.pricing_environments[position.underlying]
            total_pnl += position.get_pnl(pricing_env)
        
        return total_pnl
    
    def get_portfolio_dv01(self) -> float:
        """
        Calculate total portfolio DV01.
        
        Returns:
            Total portfolio DV01
        """
        total_dv01 = 0.0
        
        for position in self.positions.values():
            pricing_env = self.pricing_environments[position.underlying]
            total_dv01 += position.get_dv01(pricing_env)
        
        return total_dv01
    
    def get_portfolio_convexity(self) -> float:
        """
        Calculate total portfolio convexity.
        
        Returns:
            Total portfolio convexity
        """
        total_convexity = 0.0
        
        for position in self.positions.values():
            pricing_env = self.pricing_environments[position.underlying]
            total_convexity += position.get_convexity(pricing_env)
        
        return total_convexity
    
    def get_portfolio_duration(self) -> float:
        """
        Calculate portfolio weighted-average modified duration.
        
        Returns:
            Portfolio modified duration (market-value weighted)
        """
        total_value = self.get_portfolio_value()
        if total_value == 0:
            return 0.0
        
        weighted_duration = 0.0
        
        for position in self.positions.values():
            pricing_env = self.pricing_environments[position.underlying]
            mv = position.get_market_value(pricing_env)
            duration = position.get_modified_duration(pricing_env)
            weighted_duration += mv * duration
        
        return weighted_duration / total_value
    
    def get_portfolio_risk_measures(self, **kwargs) -> Dict[str, float]:
        """
        Calculate aggregated risk measures across all positions.
        
        Returns FI-specific measures: dv01, convexity, modified_duration.
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of aggregated risk measures
        """
        aggregated = {
            'market_value': 0.0,
            'dv01': 0.0,
            'convexity': 0.0,
            'modified_duration': 0.0,
        }
        
        total_value = 0.0
        weighted_duration = 0.0
        
        for position in self.positions.values():
            pricing_env = self.pricing_environments[position.underlying]
            risk_measures = position.get_risk_measures(pricing_env)
            
            mv = risk_measures.get('market_value', 0.0)
            total_value += mv
            weighted_duration += mv * risk_measures.get('modified_duration', 0.0)
            
            aggregated['dv01'] += risk_measures.get('dv01', 0.0)
            aggregated['convexity'] += risk_measures.get('convexity', 0.0)
        
        aggregated['market_value'] = total_value
        
        if total_value > 0:
            aggregated['modified_duration'] = weighted_duration / total_value
        
        return aggregated
    
    def get_risk_by_underlying(
        self,
        underlying: str
    ) -> Dict[str, float]:
        """
        Calculate aggregated risk measures for a specific underlying.
        
        Args:
            underlying: Underlying identifier
            
        Returns:
            Dictionary of aggregated risk measures for the underlying
        """
        aggregated = {
            'market_value': 0.0,
            'dv01': 0.0,
            'convexity': 0.0,
            'modified_duration': 0.0,
        }
        
        positions = self.get_positions_by_underlying(underlying)
        pricing_env = self.pricing_environments.get(underlying)
        
        if pricing_env is None:
            return aggregated
        
        total_value = 0.0
        weighted_duration = 0.0
        
        for position in positions:
            risk_measures = position.get_risk_measures(pricing_env)
            
            mv = risk_measures.get('market_value', 0.0)
            total_value += mv
            weighted_duration += mv * risk_measures.get('modified_duration', 0.0)
            
            aggregated['dv01'] += risk_measures.get('dv01', 0.0)
            aggregated['convexity'] += risk_measures.get('convexity', 0.0)
        
        aggregated['market_value'] = total_value
        
        if total_value > 0:
            aggregated['modified_duration'] = weighted_duration / total_value
        
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
            row['dirty_price'] = position.get_dirty_price(pricing_env)
            row['market_value'] = position.get_market_value(pricing_env)
            row['pnl'] = position.get_pnl(pricing_env)
            row['dv01'] = position.get_dv01(pricing_env)
            row['modified_duration'] = position.get_modified_duration(pricing_env)
            
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
                'total_dv01': 0.0,
                'portfolio_duration': 0.0,
                'long_positions': 0,
                'short_positions': 0,
                'asset_class': 'fixed_income',
            }
        
        total_value = self.get_portfolio_value()
        total_pnl = self.get_portfolio_pnl()
        total_dv01 = self.get_portfolio_dv01()
        portfolio_duration = self.get_portfolio_duration()
        
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
            'total_dv01': total_dv01,
            'portfolio_duration': portfolio_duration,
            'long_positions': long_positions,
            'short_positions': short_positions,
            'pnl_percentage': (total_pnl / (total_value - total_pnl) * 100) if (total_value - total_pnl) != 0 else 0.0,
            'asset_class': 'fixed_income',
        }
    
    def __len__(self):
        """Return number of positions in portfolio."""
        return len(self.positions)
    
    def __repr__(self):
        return (
            f"FIPortfolio(name={self.portfolio_name}, "
            f"positions={len(self.positions)}, "
            f"underlyings={len(self.pricing_environments)})"
        )

