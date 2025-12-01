"""
Enumeration types for pricing engines.
"""

from enum import Enum, auto


class AmericanAnalyticalMethod(Enum):
    """Analytical methods for American option pricing."""

    BS93 = "BS93"
    BS02 = "BS02"
    BAW = "BAW"

    def __str__(self):
        return self.value


class MonteCarloMethod(Enum):
    """Monte Carlo methods for pricing."""

    PSEUDO = "pseudo"
    QUASI = "quasi"
    RANDOMIZED_QUASI = "randomized_quasi"

    def __str__(self):
        return self.value


class EngineType(Enum):
    """Type of pricing engine."""

    ANALYTICAL = auto()
    MONTE_CARLO = auto()
    PDE = auto()
    QUADRATURE = auto()

    def __str__(self):
        return self.name.replace("_", " ").title()

    def __call__(self, method=None):
        """
        Enable syntax like EngineType.ANALYTICAL(AmericanAnalyticalMethod.BS93).

        This creates a tuple representing both the engine type and specific method,
        which can be unpacked during engine initialization.
        """
        if method is not None:
            return (self, method)
        return self
