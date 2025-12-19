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


class PDEMethod(Enum):
    """PDE finite difference methods for pricing."""

    CRANK_NICOLSON = "crank_nicolson"
    EXPLICIT_EULER = "explicit_euler"
    IMPLICIT_EULER = "implicit_euler"

    def __str__(self):
        return self.value


class ConvertibleBondMethod(Enum):
    """Methods for convertible bond pricing."""

    BINOMIAL_GS = "binomial_gs"  # Goldman Sachs credit-adjusted binomial
    TRINOMIAL_HW = "trinomial_hw"  # Hull-White trinomial with default
    JUMP_DIFFUSION = "jump_diffusion"  # Bloomberg OVCV model
    TF = "tf"  # Tsiveriotis-Fernandes decomposition

    def __str__(self):
        return self.value


class ConvertibleBondTrinomialVolScheme(Enum):
    """Volatility schemes for the trinomial convertible bond tree."""

    CONSTANT_VOL = "constant_vol"
    LOG_FIXED_DX = "log_fixed_dx"
    LOG_VARIABLE_DX = "log_variable_dx"

    def __str__(self):
        return self.value


class EngineType(Enum):
    """Type of pricing engine."""

    ANALYTICAL = auto()
    MONTE_CARLO = auto()
    PDE = auto()
    QUADRATURE = auto()
    TREE = auto()

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
