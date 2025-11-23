"""
Enumeration types for delta one product specifications.
"""
from enum import Enum, auto


class DeltaOneType(Enum):
    """Type of delta one product."""
    STOCK = auto()
    INDEX = auto()
    ETF = auto()
    FUTURES = auto()
    
    def __str__(self):
        return self.name.capitalize()

