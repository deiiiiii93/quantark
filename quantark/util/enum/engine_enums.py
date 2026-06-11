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


class AsianAnalyticalMethod(Enum):
    """Analytical methods for Asian option pricing.
    
    - KEMNA_VORST: Exact closed-form for geometric average options
    - GEOMETRIC_DISCRETE: Discrete geometric average-rate using term-structure vols
    - TURNBULL_WAKEMAN: Moment matching for arithmetic average options
    - LEVY: Alternative arithmetic approximation (requires b != 0)
    - CURRAN: Geometric conditioning approximation
    - DISCRETE_HHM: Discrete arithmetic (Haug-Haug-Margrabe)
    """

    KEMNA_VORST = "kemna_vorst"
    GEOMETRIC_DISCRETE = "geometric_discrete"
    TURNBULL_WAKEMAN = "turnbull_wakeman"
    LEVY = "levy"
    CURRAN = "curran"
    DISCRETE_HHM = "discrete_hhm"

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


class QuadratureMethod(Enum):
    """Quadrature methods for numerical integration pricing."""

    SIMPSON = "simpson"
    GAUSS_LEGENDRE = "gauss_legendre"

    def __str__(self):
        return self.value


class KnockInMonitoringMode(Enum):
    """Treatment of discrete knock-in schedules in quadrature engines.

    Attributes:
        EXACT_DISCRETE: Price every KI observation date exactly; the engine
            adaptively refines its spatial grid to resolve short diffusion
            intervals. This is the default.
        BGK_APPROXIMATION: Opt-in performance mode that replaces a dense
            discrete KI schedule with continuous Brownian-bridge monitoring
            at a Broadie-Glasserman-Kou (1997) shifted barrier. Only valid
            when the schedule is approximately regular, the resolved barrier
            is constant, monitoring covers the full pricing horizon, and the
            volatility surface is stable over the schedule; the engine raises
            ValidationError otherwise. A first-order residual bias remains
            (largest for coarse spacing and strong drift).
    """

    EXACT_DISCRETE = "exact_discrete"
    BGK_APPROXIMATION = "bgk_approximation"

    def __str__(self):
        return self.value


class GreeksCalculationMode(Enum):
    """Mode for calculating delta/gamma in GreeksCalculator.

    Controls whether delta and gamma are calculated using the engine's
    built-in calculate_greeks() method or the finite difference bump method.

    Attributes:
        ENGINE: Use engine.calculate_greeks() if available (e.g., PDE grid Greeks)
        BUMP: Use finite difference bump method (universal, but requires re-pricing)
        AUTO: Use engine method for PDE engines, bump method otherwise
    """

    ENGINE = "engine"  # Use engine.calculate_greeks() if available
    BUMP = "bump"      # Use finite difference bump method
    AUTO = "auto"      # Use engine method for PDE, bump otherwise

    def __str__(self):
        return self.value
