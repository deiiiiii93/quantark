"""
Fixed Income position class for tracking bond and bond derivative positions.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any
from datetime import datetime
import uuid

from asset.bond.product.base_bond_product import BaseBondProduct
from asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from priceenv import PricingEnvironment
from util.exceptions import ValidationError


@dataclass
class FIPosition:
    """
    Represents a single Fixed Income position in a portfolio.
    
    A position tracks a bond or bond derivative with quantity, entry details,
    and provides FI-specific risk measures (DV01, convexity, duration).
    
    Attributes:
        product: The bond product (e.g., FixedBond, BondFutures)
        quantity: Number of bonds/contracts (positive=long, negative=short)
        entry_price: Price at which position was entered (clean price)
        underlying: Identifier for the underlying/issuer
        engine: Pricing engine specific to this position
        entry_timestamp: When the position was opened
        position_id: Unique identifier for this position
        notional_per_unit: Notional per unit (default: 100 for bonds)
    """
    product: BaseBondProduct
    quantity: float
    entry_price: float
    underlying: str
    engine: Any  # BondDiscountEngine or BondFuturesEngine
    entry_timestamp: datetime
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    notional_per_unit: float = 100.0
    engine_factory: Optional[Callable[[PricingEnvironment], Any]] = None
    
    def __post_init__(self):
        """Validate position parameters."""
        self.validate()
        if self.engine_factory is None:
            engine_cls = self.engine.__class__

            def _factory(env: PricingEnvironment, cls=engine_cls):
                return cls(env)

            self.engine_factory = _factory
    
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
    
    def _engine_for_env(self, pricing_env: PricingEnvironment) -> Any:
        if self.engine_factory is not None:
            return self.engine_factory(pricing_env)
        return self.engine

    def _get_denominator(self) -> float:
        if hasattr(self.product, "get_denominator"):
            return self.product.get_denominator()
        return self.notional_per_unit

    def get_current_price(self, pricing_env: PricingEnvironment) -> float:
        """
        Get current clean price of the bond.
        
        Args:
            pricing_env: Pricing environment with rate curve
            
        Returns:
            Current clean price
        """
        valuation_date = pricing_env.valuation_date
        engine = self._engine_for_env(pricing_env)
        return engine.clean_price(self.product, valuation_date, valuation_date)
    
    def get_dirty_price(self, pricing_env: PricingEnvironment) -> float:
        """
        Get current dirty price (clean + accrued interest).
        
        Args:
            pricing_env: Pricing environment with rate curve
            
        Returns:
            Current dirty price
        """
        valuation_date = pricing_env.valuation_date
        engine = self._engine_for_env(pricing_env)
        return engine.dirty_price(self.product, valuation_date, valuation_date)
    
    def get_market_value(self, pricing_env: PricingEnvironment) -> float:
        """
        Calculate current market value of the position.
        
        Market value = dirty_price × quantity
        
        Args:
            pricing_env: Pricing environment with rate curve
            
        Returns:
            Current market value
        """
        dirty_price = self.get_dirty_price(pricing_env)
        return dirty_price * self.quantity
    
    def get_pnl(self, pricing_env: PricingEnvironment) -> float:
        """
        Calculate unrealized P&L of the position.
        
        P&L = (current_price - entry_price) × quantity
        
        Args:
            pricing_env: Pricing environment with rate curve
            
        Returns:
            Unrealized profit/loss
        """
        current_price = self.get_current_price(pricing_env)
        return (current_price - self.entry_price) * self.quantity
    
    def get_dv01(self, pricing_env: PricingEnvironment) -> float:
        """
        Calculate DV01 (dollar value of 1 basis point) for this position.
        
        DV01 = per-bond DV01 × quantity
        
        Args:
            pricing_env: Pricing environment with rate curve
            
        Returns:
            Position DV01
        """
        valuation_date = pricing_env.valuation_date
        engine = self._engine_for_env(pricing_env)
        per_unit_dv01 = engine.dv01(self.product, valuation_date, valuation_date)
        return per_unit_dv01 * self.quantity
    
    def get_modified_duration(self, pricing_env: PricingEnvironment) -> float:
        """
        Calculate modified duration.
        
        Args:
            pricing_env: Pricing environment with rate curve
            
        Returns:
            Modified duration
        """
        valuation_date = pricing_env.valuation_date
        engine = self._engine_for_env(pricing_env)
        return engine.modified_duration(self.product, valuation_date, valuation_date)
    
    def get_convexity(self, pricing_env: PricingEnvironment) -> float:
        """
        Calculate convexity for this position.
        
        Convexity measures the second-order sensitivity to rate changes.
        
        Args:
            pricing_env: Pricing environment with rate curve
            
        Returns:
            Position convexity
        """
        valuation_date = pricing_env.valuation_date
        engine = self._engine_for_env(pricing_env)
        per_unit_convexity = engine.convexity(self.product, valuation_date, valuation_date)
        market_value = self.get_market_value(pricing_env)
        return per_unit_convexity * market_value
    
    def get_risk_measures(
        self,
        pricing_env: PricingEnvironment,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate all risk measures for this position.
        
        Returns FI-specific measures: dv01, modified_duration, convexity, yield.
        
        Args:
            pricing_env: Pricing environment with rate curve
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of risk measures
        """
        valuation_date = pricing_env.valuation_date
        
        risk_measures = {
            'market_value': self.get_market_value(pricing_env),
            'dv01': self.get_dv01(pricing_env),
            'modified_duration': self.get_modified_duration(pricing_env),
            'convexity': self.get_convexity(pricing_env),
        }
        
        # Add yield if available
        try:
            current_price = self.get_current_price(pricing_env)
            engine = self._engine_for_env(pricing_env)
            ytm = engine.yield_to_maturity(
                self.product, valuation_date, current_price
            )
            risk_measures['yield'] = ytm
        except Exception:
            risk_measures['yield'] = 0.0
        
        return risk_measures

    def get_actual_notional(self) -> float:
        """
        Calculate actual notional based on denominator.

        Returns:
            Actual notional amount
        """
        return self.quantity * self._get_denominator()
    
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
            'notional_per_unit': self.notional_per_unit,
            'asset_class': 'fixed_income',
        }
    
    def __repr__(self):
        direction = "LONG" if self.is_long() else "SHORT"
        return (
            f"FIPosition(id={self.position_id[:8]}..., "
            f"{direction} {abs(self.quantity)} x {self.product.__class__.__name__}, "
            f"entry=${self.entry_price:.2f}, underlying={self.underlying})"
        )
