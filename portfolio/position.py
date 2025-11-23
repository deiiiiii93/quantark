"""
Position class for tracking individual positions in a portfolio.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.engine.base_engine import BaseEngine
from priceenv import PricingEnvironment
from asset.equity.riskmeasures import GreeksCalculator
from util.exceptions import ValidationError


@dataclass
class Position:
    """
    Represents a single position in a portfolio.
    
    A position tracks a product (e.g., option) with quantity, entry details,
    and its own pricing engine.
    
    Attributes:
        product: The derivative product (e.g., EuropeanVanillaOption)
        quantity: Number of contracts/units (positive=long, negative=short)
        entry_price: Price at which position was entered
        underlying: Identifier for the underlying asset
        engine: Pricing engine specific to this position
        entry_timestamp: When the position was opened
        position_id: Unique identifier for this position
    """
    product: BaseEquityProduct
    quantity: float
    entry_price: float
    underlying: str
    engine: BaseEngine
    entry_timestamp: datetime
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Validate position parameters."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate position parameters.
        
        Raises:
            ValidationError: If parameters are invalid
        """
        if self.quantity == 0:
            raise ValidationError("Position quantity cannot be zero")
        if self.entry_price < 0:
            raise ValidationError(f"Entry price must be non-negative, got {self.entry_price}")
        if not self.underlying:
            raise ValidationError("Underlying identifier is required")
        if self.product is None:
            raise ValidationError("Product is required")
        if self.engine is None:
            raise ValidationError("Engine is required")
    
    def get_current_price(self, pricing_env: PricingEnvironment) -> float:
        """
        Get current market price of the product.
        
        Uses the position's specific engine to price the product.
        
        Args:
            pricing_env: Pricing environment for the underlying
            
        Returns:
            Current market price
        """
        return self.engine.price(self.product, pricing_env)
    
    def get_market_value(self, pricing_env: PricingEnvironment) -> float:
        """
        Calculate current market value of the position.
        
        Market value = current_price × quantity
        
        Args:
            pricing_env: Pricing environment for the underlying
            
        Returns:
            Current market value
        """
        current_price = self.get_current_price(pricing_env)
        return current_price * self.quantity
    
    def get_pnl(self, pricing_env: PricingEnvironment) -> float:
        """
        Calculate unrealized P&L of the position.
        
        P&L = (current_price - entry_price) × quantity
        
        Args:
            pricing_env: Pricing environment for the underlying
            
        Returns:
            Unrealized profit/loss
        """
        current_price = self.get_current_price(pricing_env)
        return (current_price - self.entry_price) * self.quantity
    
    def get_greeks(
        self,
        pricing_env: PricingEnvironment,
        greeks_calculator: GreeksCalculator,
        use_analytical: bool = True
    ) -> Dict[str, float]:
        """
        Calculate Greeks for this position.
        
        Greeks are scaled by position quantity.
        
        Args:
            pricing_env: Pricing environment for the underlying
            greeks_calculator: Greeks calculator instance
            use_analytical: If True, use analytical Greeks when possible
            
        Returns:
            Dictionary of Greeks scaled by quantity
        """
        # Calculate single-contract Greeks
        if use_analytical:
            try:
                greeks = greeks_calculator.calculate_analytical_greeks(
                    self.product,
                    pricing_env
                )
            except (ValidationError, AttributeError):
                # Fall back to numerical if analytical not available
                greeks = greeks_calculator.calculate_numerical_greeks(
                    self.product,
                    pricing_env,
                    self.engine
                )
        else:
            greeks = greeks_calculator.calculate_numerical_greeks(
                self.product,
                pricing_env,
                self.engine
            )
        
        # Scale Greeks by quantity
        scaled_greeks = {}
        for key, value in greeks.items():
            if key == 'price':
                # For price, use market value instead
                scaled_greeks['market_value'] = value * self.quantity
            else:
                scaled_greeks[key] = value * self.quantity
        
        return scaled_greeks
    
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0
    
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize position to dictionary.
        
        Returns:
            Dictionary representation of position
        """
        return {
            'position_id': self.position_id,
            'underlying': self.underlying,
            'product_type': self.product.__class__.__name__,
            'product_repr': repr(self.product),
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_timestamp': self.entry_timestamp.isoformat(),
            'engine_type': self.engine.__class__.__name__,
            'direction': 'LONG' if self.is_long() else 'SHORT',
        }
    
    def __repr__(self):
        direction = "LONG" if self.is_long() else "SHORT"
        return (
            f"Position(id={self.position_id[:8]}..., "
            f"{direction} {abs(self.quantity)} x {self.product.__class__.__name__}, "
            f"entry=${self.entry_price:.2f}, underlying={self.underlying})"
        )

