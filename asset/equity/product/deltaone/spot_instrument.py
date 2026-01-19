"""
Spot instrument implementation for stocks, indices, and ETFs.
"""

import math
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from .base_deltaone_product import BaseDeltaOneProduct
from util.enum.deltaone_enums import DeltaOneType
from util.exceptions import ValidationError


@dataclass
class SpotInstrument(BaseDeltaOneProduct):
    """
    Spot instrument (Stock, Index, or ETF).
    
    Spot instruments are perpetual holdings with no maturity date.
    Their value follows the underlying asset price with delta = 1.
    
    Forward pricing follows cost-of-carry model:
        F(t,T) = S(t) * exp((r - q) * (T - t))
    
    Attributes:
        underlying: Identifier for the underlying asset (e.g., ticker symbol)
        deltaone_type: Type of spot instrument (STOCK, INDEX, or ETF)
    
    Example:
        >>> stock = SpotInstrument(underlying="AAPL", deltaone_type=DeltaOneType.STOCK)
        >>> etf = SpotInstrument(underlying="SPY", deltaone_type=DeltaOneType.ETF)
    """
    
    def __init__(
        self,
        underlying: str,
        deltaone_type: DeltaOneType,
    ):
        """
        Initialize spot instrument.
        
        Args:
            underlying: Identifier for the underlying asset
            deltaone_type: Type of instrument (STOCK, INDEX, or ETF)
        
        Raises:
            ValidationError: If deltaone_type is FUTURES (use Futures class instead)
        """
        if deltaone_type == DeltaOneType.FUTURES:
            raise ValidationError(
                "Use Futures class for futures products, not SpotInstrument"
            )
        
        # Spot instruments are perpetual (no maturity)
        super().__init__(
            underlying=underlying,
            deltaone_type=deltaone_type,
            maturity=None,
            maturity_date=None,
        )
    
    def get_forward_price(
        self,
        spot: float,
        rate: float,
        div_yield: float,
        time_to_maturity: float
    ) -> float:
        """
        Calculate the forward price using cost-of-carry model.
        
        Forward price: F(t,T) = S(t) * exp((r - q) * (T - t))
        
        Args:
            spot: Current spot price
            rate: Risk-free rate (continuously compounded)
            div_yield: Dividend yield (continuously compounded)
            time_to_maturity: Time to forward date in years
        
        Returns:
            Forward price
        
        Raises:
            ValidationError: If inputs are invalid
        """
        if spot <= 0:
            raise ValidationError(f"Spot price must be positive, got {spot}")
        if time_to_maturity < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {time_to_maturity}")
        
        # Cost-of-carry forward pricing
        carry_cost = (rate - div_yield) * time_to_maturity
        forward_price = spot * math.exp(carry_cost)
        
        return forward_price
    
    def get_current_value(self, spot: float) -> float:
        """
        Get the current value of the spot instrument.
        
        For spot instruments, current value equals spot price.
        
        Args:
            spot: Current spot price
        
        Returns:
            Current value (equals spot)
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")
        return spot

    @property
    def is_linear(self) -> bool:
        return True
    
    def __repr__(self):
        return f"SpotInstrument({self.underlying}, {self.deltaone_type})"
