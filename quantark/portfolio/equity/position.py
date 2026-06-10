"""
Equity position class for tracking individual equity derivative positions.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.engine.base_engine import BaseEngine
from priceenv import PricingEnvironment
from asset.equity.riskmeasures import GreeksCalculator
from cashleg import CashLeg, LegPV, TradeValueBreakdown, value_leg
from util.exceptions import ValidationError


@dataclass
class EquityPosition:
    """
    Represents a single equity derivative position in a portfolio.
    
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
        cash_legs: Optional cash terms attached to this position
    """
    product: BaseEquityProduct
    quantity: float
    entry_price: float
    underlying: str
    engine: BaseEngine
    entry_timestamp: datetime
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cash_legs: List[CashLeg] = field(default_factory=list)
    
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

    def get_trade_value(self, pricing_env: PricingEnvironment) -> float:
        """
        Calculate full trade value including product and attached cash legs.

        For positions without cash legs, this is exactly get_market_value().
        """
        if not self.cash_legs:
            return self.get_market_value(pricing_env)

        needs_distribution = any(
            leg.requires_event_distribution() for leg in self.cash_legs
        )
        result = self.engine.price_with_events(
            self.product, pricing_env, emit_distribution=needs_distribution
        )
        unit_notional = self._get_unit_notional(pricing_env)
        leg_pv_total = sum(
            value_leg(leg, result.event_distribution, pricing_env, unit_notional)
            for leg in self.cash_legs
        )
        return (result.npv + leg_pv_total) * self.quantity

    def get_trade_value_breakdown(
        self, pricing_env: PricingEnvironment
    ) -> TradeValueBreakdown:
        """Return product and per-leg PV attribution."""
        if not self.cash_legs:
            return TradeValueBreakdown(
                product_npv=self.get_market_value(pricing_env),
                leg_pvs={},
            )

        needs_distribution = any(
            leg.requires_event_distribution() for leg in self.cash_legs
        )
        result = self.engine.price_with_events(
            self.product, pricing_env, emit_distribution=needs_distribution
        )
        unit_notional = self._get_unit_notional(pricing_env)
        leg_pvs = {}
        for leg in self.cash_legs:
            pv = (
                value_leg(leg, result.event_distribution, pricing_env, unit_notional)
                * self.quantity
            )
            leg_pvs[leg.leg_id] = LegPV(
                name=leg.name,
                direction=leg.direction,
                pv=pv,
            )

        return TradeValueBreakdown(
            product_npv=result.npv * self.quantity,
            leg_pvs=leg_pvs,
        )

    def get_trade_greeks(
        self,
        pricing_env: PricingEnvironment,
        greeks_calculator: GreeksCalculator,
    ) -> Dict[str, float]:
        """Calculate trade-level Greeks by bumping get_trade_value()."""
        from copy import deepcopy
        from param import FlatRateCurve

        bump = self.engine.params.bump_size
        base_value = self.get_trade_value(pricing_env)

        env_up = deepcopy(pricing_env)
        env_up.spot_quote.spot *= 1.0 + bump
        env_down = deepcopy(pricing_env)
        env_down.spot_quote.spot *= 1.0 - bump
        value_up = self.get_trade_value(env_up)
        value_down = self.get_trade_value(env_down)

        spot_bump_amount = pricing_env.spot * bump
        delta = (value_up - value_down) / (2.0 * spot_bump_amount)
        gamma = (value_up - 2.0 * base_value + value_down) / (spot_bump_amount**2)

        env_vol_up = deepcopy(pricing_env)
        env_vol_down = deepcopy(pricing_env)
        self._bump_flat_or_term_vol(env_vol_up, bump)
        self._bump_flat_or_term_vol(env_vol_down, -bump)
        vega = (
            self.get_trade_value(env_vol_up) - self.get_trade_value(env_vol_down)
        ) / (2.0 * bump)

        current_rate = pricing_env.get_rate(self.product.get_maturity(pricing_env))
        env_rate_up = deepcopy(pricing_env)
        env_rate_down = deepcopy(pricing_env)
        env_rate_up.rate_curve = FlatRateCurve(current_rate + bump)
        env_rate_down.rate_curve = FlatRateCurve(current_rate - bump)
        rho = (
            self.get_trade_value(env_rate_up) - self.get_trade_value(env_rate_down)
        ) / (2.0 * bump)

        return {
            "price": base_value,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "rho": rho,
        }

    def get_actual_notional(
        self, pricing_env: Optional[PricingEnvironment] = None
    ) -> float:
        """
        Calculate actual notional based on contract multiplier.

        Uses product.initial_price when available; otherwise falls back to spot
        from pricing_env.

        Args:
            pricing_env: Pricing environment for spot fallback (optional)

        Returns:
            Actual notional amount
        """
        base_price = getattr(self.product, "initial_price", 0.0) or 0.0
        if base_price <= 0:
            if pricing_env is None:
                raise ValidationError(
                    "PricingEnvironment required when product.initial_price is not set"
                )
            base_price = pricing_env.spot

        contract_multiplier = getattr(self.product, "contract_multiplier", 1.0)
        return self.quantity * base_price * contract_multiplier

    def _get_unit_notional(self, pricing_env: PricingEnvironment) -> float:
        """Return per-unit notional so quantity scales product and leg PV once."""
        return self.get_actual_notional(pricing_env) / self.quantity

    @staticmethod
    def _bump_flat_or_term_vol(pricing_env: PricingEnvironment, bump: float) -> None:
        vol_surface = pricing_env.vol_surface
        if hasattr(vol_surface, "volatility"):
            vol_surface.volatility += bump
            return
        if hasattr(vol_surface, "vols"):
            vol_surface.vols = [float(vol) + bump for vol in vol_surface.vols]
            return
        raise ValidationError(
            f"Unsupported volatility surface bump for {type(vol_surface).__name__}"
        )
    
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
    
    def get_risk_measures(
        self,
        pricing_env: PricingEnvironment,
        greeks_calculator: Optional[GreeksCalculator] = None,
        use_analytical: bool = True
    ) -> Dict[str, float]:
        """
        Calculate risk measures for this position.
        
        For equity positions, this returns Greeks (delta, gamma, vega, theta, rho).
        
        Args:
            pricing_env: Pricing environment for the underlying
            greeks_calculator: Greeks calculator instance (optional)
            use_analytical: If True, use analytical Greeks when possible
            
        Returns:
            Dictionary of risk measures
        """
        if greeks_calculator is None:
            greeks_calculator = GreeksCalculator()
        return self.get_greeks(pricing_env, greeks_calculator, use_analytical)
    
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
            'asset_class': 'equity',
        }
    
    def __repr__(self):
        direction = "LONG" if self.is_long() else "SHORT"
        return (
            f"EquityPosition(id={self.position_id[:8]}..., "
            f"{direction} {abs(self.quantity)} x {self.product.__class__.__name__}, "
            f"entry=${self.entry_price:.2f}, underlying={self.underlying})"
        )


# Backward compatibility alias
Position = EquityPosition
