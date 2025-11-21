"""
Dividend yield representations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from util.exceptions import ValidationError


class DividendYield(ABC):
    """
    Abstract base class for dividend yields.
    
    Dividend yields can be continuous or discrete.
    """
    
    @abstractmethod
    def get_yield(self, time_to_maturity: float) -> float:
        """
        Get dividend yield for given maturity.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Annualized dividend yield (continuously compounded)
        """
        pass


@dataclass
class ContinuousDividendYield(DividendYield):
    """
    Continuous dividend yield.
    
    Models dividends as a continuous yield, suitable for index options
    or stocks with frequent dividend payments.
    
    Attributes:
        div_yield: Constant annualized dividend yield (continuously compounded)
    """
    div_yield: float
    
    def __post_init__(self):
        """Validate dividend yield."""
        if self.div_yield < 0:
            raise ValidationError(f"Dividend yield must be non-negative, got {self.div_yield}")
        if self.div_yield > 0.20:  # 20% div yield - sanity check
            raise ValidationError(f"Dividend yield seems unreasonably high: {self.div_yield}")
    
    def get_yield(self, time_to_maturity: float) -> float:
        """
        Return constant dividend yield.
        
        Args:
            time_to_maturity: Time to maturity (ignored)
            
        Returns:
            Constant dividend yield
        """
        return self.div_yield
    
    def __repr__(self):
        return f"ContinuousDividendYield(yield={self.div_yield:.2%})"


@dataclass
class NoDividend(DividendYield):
    """
    Zero dividend yield.
    
    Convenience class for stocks that pay no dividends.
    """
    
    def get_yield(self, time_to_maturity: float) -> float:
        """
        Return zero dividend yield.
        
        Args:
            time_to_maturity: Time to maturity (ignored)
            
        Returns:
            Zero
        """
        return 0.0
    
    def __repr__(self):
        return "NoDividend()"

