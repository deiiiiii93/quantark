"""
Base class for delta one products.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from ..base_equity_product import BaseEquityProduct
from util.enum.deltaone_enums import DeltaOneType
from util.exceptions import ValidationError


@dataclass
class BaseDeltaOneProduct(BaseEquityProduct):
    """
    Abstract base class for delta one products.
    
    Delta one products have a delta of approximately 1.0 with respect to their
    underlying asset. This includes stocks, indices, ETFs, and futures.
    
    Attributes:
        underlying: Identifier for the underlying asset (e.g., ticker symbol)
        deltaone_type: Type of delta one product (STOCK, INDEX, ETF, FUTURES)
        maturity: Time to maturity in years (None for perpetual instruments like stocks)
        maturity_date: Optional date when product matures (for futures)
    """
    
    underlying: str
    deltaone_type: DeltaOneType
    maturity: Optional[float] = None
    maturity_date: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate product parameters."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate delta one product parameters.
        
        Raises:
            ValidationError: If parameters are invalid
        """
        if not self.underlying:
            raise ValidationError("Underlying identifier is required")
        
        if not isinstance(self.deltaone_type, DeltaOneType):
            raise ValidationError(f"Invalid delta one type: {self.deltaone_type}")
        
        # Validate maturity specifications
        has_maturity = self.maturity is not None and self.maturity > 0
        has_maturity_date = self.maturity_date is not None
        
        if has_maturity and has_maturity_date:
            raise ValidationError("Cannot provide both maturity and maturity_date")
        
        if has_maturity and self.maturity < 0:
            raise ValidationError(f"Maturity must be non-negative, got {self.maturity}")
    
    def get_maturity(self, pricing_env=None) -> float:
        """
        Get time to maturity in years.
        
        For perpetual instruments (stocks, indices, ETFs), returns a very large number
        to represent infinite maturity. For futures, calculates maturity from dates
        or returns the maturity float.
        
        Args:
            pricing_env: Pricing environment (required if using date-based calculation)
        
        Returns:
            Time to maturity in years
            
        Raises:
            ValidationError: If date-based but no pricing_env provided
        """
        # Perpetual instruments (stocks, indices, ETFs)
        if self.maturity is None and self.maturity_date is None:
            return 1e10  # Very large number representing infinity
        
        # Date-based maturity
        if self.maturity_date is not None:
            if pricing_env is None:
                raise ValidationError(
                    "PricingEnvironment required for date-based maturity calculation"
                )
            from util.calendar import calculate_year_fraction
            
            # Validate valuation date is before maturity date
            if pricing_env.valuation_date >= self.maturity_date:
                raise ValidationError(
                    f"Valuation date ({pricing_env.valuation_date}) must be before "
                    f"maturity date ({self.maturity_date})"
                )
            
            return calculate_year_fraction(
                pricing_env.valuation_date,
                self.maturity_date,
                pricing_env.day_count_convention,
                pricing_env.bus_days_in_year
            )
        else:
            # Float-based maturity
            return self.maturity
    
    def get_payoff(self, spot: float) -> float:
        """
        Calculate the payoff given a spot price.
        
        For delta one products, the payoff is simply the spot price
        (or spot - 0 for long position).
        
        Args:
            spot: Spot price at evaluation
        
        Returns:
            Payoff value (equals spot for delta one)
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")
        return spot
    
    @abstractmethod
    def get_forward_price(self, spot: float, rate: float, div_yield: float, time_to_maturity: float) -> float:
        """
        Calculate the forward price of the product.
        
        Args:
            spot: Current spot price
            rate: Risk-free rate
            div_yield: Dividend yield
            time_to_maturity: Time to maturity in years
        
        Returns:
            Forward price
        """
        pass
    
    def __repr__(self):
        if self.maturity_date:
            return f"{self.__class__.__name__}({self.underlying}, {self.deltaone_type}, maturity_date={self.maturity_date})"
        elif self.maturity is not None:
            return f"{self.__class__.__name__}({self.underlying}, {self.deltaone_type}, T={self.maturity:.4f})"
        else:
            return f"{self.__class__.__name__}({self.underlying}, {self.deltaone_type})"

