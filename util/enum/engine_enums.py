"""
Enumeration types for pricing engines.
"""
from enum import Enum, auto


class EngineType(Enum):
    """Type of pricing engine."""
    ANALYTICAL = auto()
    MONTE_CARLO = auto()
    PDE = auto()
    QUADRATURE = auto()
    
    def __str__(self):
        return self.name.replace('_', ' ').title()

