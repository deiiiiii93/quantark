"""
Risk-free rate curve representations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from util.exceptions import ValidationError


class RateCurve(ABC):
    """
    Abstract base class for risk-free rate curves.
    
    Rate curves provide risk-free interest rates as a function of maturity.
    """
    
    @abstractmethod
    def get_rate(self, time_to_maturity: float) -> float:
        """
        Get risk-free rate for given maturity.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Annualized risk-free rate (continuously compounded)
        """
        pass
    
    @abstractmethod
    def get_discount_factor(self, time_to_maturity: float) -> float:
        """
        Get discount factor for given maturity.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Discount factor exp(-r*T)
        """
        pass


@dataclass
class FlatRateCurve(RateCurve):
    """
    Flat (constant) rate curve.
    
    Returns the same rate regardless of maturity.
    
    Attributes:
        rate: Constant annualized risk-free rate (continuously compounded)
    """
    rate: float
    
    def __post_init__(self):
        """Validate rate - allow negative rates but warn if extreme."""
        if self.rate < -0.10:  # -10% rate - sanity check
            raise ValidationError(f"Rate seems unreasonably low: {self.rate}")
        if self.rate > 0.50:  # 50% rate - sanity check
            raise ValidationError(f"Rate seems unreasonably high: {self.rate}")
    
    def get_rate(self, time_to_maturity: float) -> float:
        """
        Return constant rate.
        
        Args:
            time_to_maturity: Time to maturity (ignored)
            
        Returns:
            Constant risk-free rate
        """
        return self.rate
    
    def get_discount_factor(self, time_to_maturity: float) -> float:
        """
        Calculate discount factor.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Discount factor exp(-r*T)
        """
        import math
        if time_to_maturity < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {time_to_maturity}")
        return math.exp(-self.rate * time_to_maturity)
    
    def __repr__(self):
        return f"FlatRateCurve(rate={self.rate:.2%})"

