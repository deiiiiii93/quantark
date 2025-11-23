"""
Transaction cost models for backtest execution.

Provides various models for calculating trading costs including commissions,
slippage, and bid-ask spreads.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import math
from util.exceptions import ValidationError


class TransactionCostModel(ABC):
    """
    Abstract base class for transaction cost models.
    
    Subclasses should implement the calculate_cost method to compute
    the total cost of executing a trade.
    """
    
    @abstractmethod
    def calculate_cost(
        self,
        quantity: float,
        price: float,
        notional: float,
        instrument_type: str,
        trade_type: str,
        **kwargs
    ) -> float:
        """
        Calculate transaction cost for a trade.
        
        Args:
            quantity: Quantity traded (positive or negative)
            price: Execution price per unit
            notional: Total notional value (abs(quantity) * price)
            instrument_type: Type of instrument ('spot', 'futures', 'option')
            trade_type: Type of trade ('open', 'close', 'hedge')
            **kwargs: Additional parameters
            
        Returns:
            Total transaction cost (always positive)
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {}


class ZeroCostModel(TransactionCostModel):
    """
    Zero cost model for frictionless trading.
    
    Always returns zero cost - useful for theoretical analysis
    or when transaction costs are negligible.
    """
    
    def calculate_cost(
        self,
        quantity: float,
        price: float,
        notional: float,
        instrument_type: str,
        trade_type: str,
        **kwargs
    ) -> float:
        """Return zero cost."""
        return 0.0
    
    def __repr__(self) -> str:
        return "ZeroCostModel()"


class FixedCostModel(TransactionCostModel):
    """
    Fixed commission per trade.
    
    Charges a flat fee regardless of trade size.
    
    Attributes:
        commission_per_trade: Fixed cost per trade
    """
    
    def __init__(self, commission_per_trade: float):
        """
        Initialize fixed cost model.
        
        Args:
            commission_per_trade: Fixed cost per trade (must be non-negative)
            
        Raises:
            ValidationError: If commission is negative
        """
        if commission_per_trade < 0:
            raise ValidationError(
                f"Commission must be non-negative, got {commission_per_trade}"
            )
        self.commission_per_trade = commission_per_trade
    
    def calculate_cost(
        self,
        quantity: float,
        price: float,
        notional: float,
        instrument_type: str,
        trade_type: str,
        **kwargs
    ) -> float:
        """Return fixed commission."""
        return self.commission_per_trade
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {'commission_per_trade': self.commission_per_trade}
    
    def __repr__(self) -> str:
        return f"FixedCostModel(commission={self.commission_per_trade})"


class ProportionalCostModel(TransactionCostModel):
    """
    Proportional commission based on notional value.
    
    Charges a percentage of the trade notional.
    
    Attributes:
        commission_rate: Commission rate (e.g., 0.001 = 10 bps = 0.1%)
    """
    
    def __init__(self, commission_rate: float):
        """
        Initialize proportional cost model.
        
        Args:
            commission_rate: Commission rate as decimal (e.g., 0.001 = 0.1%)
            
        Raises:
            ValidationError: If rate is negative or unreasonably high
        """
        if commission_rate < 0:
            raise ValidationError(
                f"Commission rate must be non-negative, got {commission_rate}"
            )
        if commission_rate > 0.1:  # 10% seems unreasonable
            raise ValidationError(
                f"Commission rate {commission_rate:.2%} seems unreasonably high"
            )
        self.commission_rate = commission_rate
    
    def calculate_cost(
        self,
        quantity: float,
        price: float,
        notional: float,
        instrument_type: str,
        trade_type: str,
        **kwargs
    ) -> float:
        """Calculate proportional commission."""
        return abs(notional) * self.commission_rate
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {'commission_rate': self.commission_rate}
    
    def __repr__(self) -> str:
        return f"ProportionalCostModel(rate={self.commission_rate:.4f})"


class SlippageModel(TransactionCostModel):
    """
    Market impact slippage model.
    
    Models the price impact of large trades using either:
    - Linear model: impact = k * |quantity|
    - Square-root model: impact = k * sqrt(|quantity|)
    
    Attributes:
        impact_coefficient: Slippage coefficient
        model_type: 'linear' or 'sqrt' (square-root)
    """
    
    def __init__(self, impact_coefficient: float, model_type: str = 'linear'):
        """
        Initialize slippage model.
        
        Args:
            impact_coefficient: Impact coefficient (typically small, e.g., 0.001)
            model_type: 'linear' or 'sqrt' for impact scaling
            
        Raises:
            ValidationError: If parameters are invalid
        """
        if impact_coefficient < 0:
            raise ValidationError(
                f"Impact coefficient must be non-negative, got {impact_coefficient}"
            )
        if model_type not in ['linear', 'sqrt']:
            raise ValidationError(
                f"Model type must be 'linear' or 'sqrt', got '{model_type}'"
            )
        
        self.impact_coefficient = impact_coefficient
        self.model_type = model_type
    
    def calculate_cost(
        self,
        quantity: float,
        price: float,
        notional: float,
        instrument_type: str,
        trade_type: str,
        **kwargs
    ) -> float:
        """Calculate slippage cost based on market impact."""
        abs_quantity = abs(quantity)
        
        if self.model_type == 'linear':
            impact_fraction = self.impact_coefficient * abs_quantity
        else:  # sqrt
            impact_fraction = self.impact_coefficient * math.sqrt(abs_quantity)
        
        # Slippage cost as fraction of notional
        slippage_cost = abs(notional) * impact_fraction
        
        return slippage_cost
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'impact_coefficient': self.impact_coefficient,
            'model_type': self.model_type
        }
    
    def __repr__(self) -> str:
        return f"SlippageModel(coef={self.impact_coefficient}, type={self.model_type})"


class BidAskSpreadModel(TransactionCostModel):
    """
    Bid-ask spread cost model.
    
    Models the cost of crossing the spread. Assumes:
    - Buys execute at ask = mid * (1 + spread/2)
    - Sells execute at bid = mid * (1 - spread/2)
    
    Attributes:
        spread_bps: Bid-ask spread in basis points (e.g., 10 = 10 bps = 0.1%)
    """
    
    def __init__(self, spread_bps: float):
        """
        Initialize bid-ask spread model.
        
        Args:
            spread_bps: Spread in basis points (e.g., 10 = 0.1%)
            
        Raises:
            ValidationError: If spread is negative
        """
        if spread_bps < 0:
            raise ValidationError(
                f"Spread must be non-negative, got {spread_bps}"
            )
        self.spread_bps = spread_bps
        self.spread_fraction = spread_bps / 10000  # Convert bps to decimal
    
    def calculate_cost(
        self,
        quantity: float,
        price: float,
        notional: float,
        instrument_type: str,
        trade_type: str,
        **kwargs
    ) -> float:
        """Calculate cost of crossing bid-ask spread."""
        # Half-spread cost on notional
        cost = abs(notional) * (self.spread_fraction / 2)
        return cost
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {'spread_bps': self.spread_bps}
    
    def __repr__(self) -> str:
        return f"BidAskSpreadModel(spread={self.spread_bps} bps)"


class CompleteCostModel(TransactionCostModel):
    """
    Complete cost model combining multiple cost components.
    
    Combines:
    - Fixed commission
    - Proportional commission
    - Slippage
    - Bid-ask spread
    
    All components are optional and can be disabled by setting to zero.
    
    Attributes:
        fixed_commission: Fixed cost per trade
        proportional_rate: Proportional commission rate
        slippage_coefficient: Slippage impact coefficient
        slippage_type: 'linear' or 'sqrt'
        spread_bps: Bid-ask spread in basis points
    """
    
    def __init__(
        self,
        fixed_commission: float = 0.0,
        proportional_rate: float = 0.0,
        slippage_coefficient: float = 0.0,
        slippage_type: str = 'linear',
        spread_bps: float = 0.0
    ):
        """
        Initialize complete cost model.
        
        Args:
            fixed_commission: Fixed cost per trade
            proportional_rate: Proportional commission rate
            slippage_coefficient: Slippage impact coefficient
            slippage_type: 'linear' or 'sqrt'
            spread_bps: Bid-ask spread in basis points
            
        Raises:
            ValidationError: If any parameters are invalid
        """
        if fixed_commission < 0:
            raise ValidationError(
                f"Fixed commission must be non-negative, got {fixed_commission}"
            )
        if proportional_rate < 0:
            raise ValidationError(
                f"Proportional rate must be non-negative, got {proportional_rate}"
            )
        if slippage_coefficient < 0:
            raise ValidationError(
                f"Slippage coefficient must be non-negative, got {slippage_coefficient}"
            )
        if slippage_type not in ['linear', 'sqrt']:
            raise ValidationError(
                f"Slippage type must be 'linear' or 'sqrt', got '{slippage_type}'"
            )
        if spread_bps < 0:
            raise ValidationError(
                f"Spread must be non-negative, got {spread_bps}"
            )
        
        self.fixed_commission = fixed_commission
        self.proportional_rate = proportional_rate
        self.slippage_coefficient = slippage_coefficient
        self.slippage_type = slippage_type
        self.spread_bps = spread_bps
        
        # Create component models
        self._fixed_model = FixedCostModel(fixed_commission) if fixed_commission > 0 else None
        self._prop_model = ProportionalCostModel(proportional_rate) if proportional_rate > 0 else None
        self._slippage_model = SlippageModel(slippage_coefficient, slippage_type) if slippage_coefficient > 0 else None
        self._spread_model = BidAskSpreadModel(spread_bps) if spread_bps > 0 else None
    
    def calculate_cost(
        self,
        quantity: float,
        price: float,
        notional: float,
        instrument_type: str,
        trade_type: str,
        **kwargs
    ) -> float:
        """Calculate total cost from all components."""
        total_cost = 0.0
        
        if self._fixed_model:
            total_cost += self._fixed_model.calculate_cost(
                quantity, price, notional, instrument_type, trade_type, **kwargs
            )
        
        if self._prop_model:
            total_cost += self._prop_model.calculate_cost(
                quantity, price, notional, instrument_type, trade_type, **kwargs
            )
        
        if self._slippage_model:
            total_cost += self._slippage_model.calculate_cost(
                quantity, price, notional, instrument_type, trade_type, **kwargs
            )
        
        if self._spread_model:
            total_cost += self._spread_model.calculate_cost(
                quantity, price, notional, instrument_type, trade_type, **kwargs
            )
        
        return total_cost
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'fixed_commission': self.fixed_commission,
            'proportional_rate': self.proportional_rate,
            'slippage_coefficient': self.slippage_coefficient,
            'slippage_type': self.slippage_type,
            'spread_bps': self.spread_bps
        }
    
    def get_cost_breakdown(
        self,
        quantity: float,
        price: float,
        notional: float,
        instrument_type: str,
        trade_type: str,
        **kwargs
    ) -> Dict[str, float]:
        """
        Get breakdown of cost components.
        
        Args:
            quantity: Quantity traded
            price: Execution price
            notional: Total notional
            instrument_type: Instrument type
            trade_type: Trade type
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with cost breakdown by component
        """
        breakdown = {
            'fixed': 0.0,
            'proportional': 0.0,
            'slippage': 0.0,
            'spread': 0.0,
            'total': 0.0
        }
        
        if self._fixed_model:
            breakdown['fixed'] = self._fixed_model.calculate_cost(
                quantity, price, notional, instrument_type, trade_type, **kwargs
            )
        
        if self._prop_model:
            breakdown['proportional'] = self._prop_model.calculate_cost(
                quantity, price, notional, instrument_type, trade_type, **kwargs
            )
        
        if self._slippage_model:
            breakdown['slippage'] = self._slippage_model.calculate_cost(
                quantity, price, notional, instrument_type, trade_type, **kwargs
            )
        
        if self._spread_model:
            breakdown['spread'] = self._spread_model.calculate_cost(
                quantity, price, notional, instrument_type, trade_type, **kwargs
            )
        
        breakdown['total'] = sum(breakdown.values())
        
        return breakdown
    
    def __repr__(self) -> str:
        components = []
        if self.fixed_commission > 0:
            components.append(f"fixed={self.fixed_commission}")
        if self.proportional_rate > 0:
            components.append(f"prop={self.proportional_rate:.4f}")
        if self.slippage_coefficient > 0:
            components.append(f"slip={self.slippage_coefficient:.4f}")
        if self.spread_bps > 0:
            components.append(f"spread={self.spread_bps}bps")
        
        if not components:
            return "CompleteCostModel(zero)"
        
        return f"CompleteCostModel({', '.join(components)})"

