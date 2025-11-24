"""
Stress type definitions and enumerations.
"""

from enum import Enum
from typing import Any


class StressType(Enum):
    """
    Type of stress to apply to a parameter.
    
    Attributes:
        ABSOLUTE: Add/subtract an absolute value to the parameter
        PERCENTAGE: Apply a percentage change to the parameter
        VALUE: Set parameter to a specific value
    """
    ABSOLUTE = "absolute"
    PERCENTAGE = "percentage"
    VALUE = "value"
    
    def __str__(self):
        return self.value
    
    def apply(self, current_value: float, stress_value: float) -> float:
        """
        Apply stress to a current value.
        
        Args:
            current_value: Current value of the parameter
            stress_value: Stress magnitude
            
        Returns:
            Stressed value
            
        Examples:
            >>> StressType.ABSOLUTE.apply(100, 10)  # Add 10
            110
            >>> StressType.PERCENTAGE.apply(100, 0.1)  # +10%
            110.0
            >>> StressType.PERCENTAGE.apply(100, -0.2)  # -20%
            80.0
            >>> StressType.VALUE.apply(100, 50)  # Set to 50
            50
        """
        if self == StressType.ABSOLUTE:
            return current_value + stress_value
        elif self == StressType.PERCENTAGE:
            return current_value * (1.0 + stress_value)
        elif self == StressType.VALUE:
            return stress_value
        else:
            raise ValueError(f"Unknown stress type: {self}")


class StressLevel(Enum):
    """
    Level at which to apply stress.
    
    Attributes:
        PORTFOLIO: Apply stress to all pricing environments in portfolio
        UNDERLYING: Apply stress to specific underlying's pricing environment
        POSITION: Apply stress to specific position's pricing environment
    """
    PORTFOLIO = "portfolio"
    UNDERLYING = "underlying"
    POSITION = "position"
    
    def __str__(self):
        return self.value


class StressableParameter(Enum):
    """
    Parameters that can be stressed in a PricingEnvironment.
    
    This is a reference enum - the actual stressing is flexible and can
    target any parameter in PricingEnvironment.
    """
    SPOT = "spot"
    VOLATILITY = "volatility"
    RATE = "rate"
    DIVIDEND_YIELD = "dividend_yield"
    
    # For more complex scenarios
    VOL_SURFACE = "vol_surface"
    RATE_CURVE = "rate_curve"
    
    def __str__(self):
        return self.value

