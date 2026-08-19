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


class FxRangeAccrualMethod(Enum):
    """Per-observation digital pricing method for FX range accrual engines.

    Both methods price the same range digital ``P(L < X_t < U)``; they differ
    only in how the digital is evaluated against the volatility surface:

    - DIGITAL_COMBINATION: closed-form ``N(d2_L) - N(d2_U)`` using the implied
      vol read from the surface at each barrier strike. Exact under a flat /
      Black-Scholes surface; under a smile it is the *sticky-strike,
      level-only* approximation (it ignores the ``-vega * dσ/dK`` skew term).
    - CALL_PUT_SPREAD: static replication ``[C(K-h) - C(K+h)] / (2h)`` with the
      two vanilla legs priced through the surface, so the smile slope is
      captured. Converges to DIGITAL_COMBINATION as ``h -> 0`` under flat vol.
    """

    DIGITAL_COMBINATION = "digital_combination"
    CALL_PUT_SPREAD = "call_put_spread"

    def __str__(self):
        return self.value


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


class EventProjectionMode(Enum):
    """Spatial representation of discrete autocallable event operators.

    Attributes:
        NODAL: Legacy behavior — a discrete coupon/KO/KI transition uses
            whole-node masks or the engine's configured nodal smoothing.
            Assigning an entire trigger cell to one branch can displace the
            effective trigger by up to half a cell.
        CELL_AVERAGE: Conservative finite-volume projection — each node
            receives the exact dual-cell average of the piecewise-linear
            event jump, so the cell straddling the threshold is split by its
            actual overlap. Applies to discretely monitored PDE and QUAD
            events; continuously monitored barriers keep their native
            treatment.
    """

    NODAL = "nodal"
    CELL_AVERAGE = "cell_average"

    def __str__(self):
        return self.value


class ContinuousKICorrection(Enum):
    """Temporal treatment of continuously monitored knock-in in PDE solvers.

    Attributes:
        FIRST_PASSAGE: Default. Per-step application of the KI regime jump is
            discrete monitoring at the time-step width, which misses barrier
            crossings inside a step and biases the PV high by O(sqrt(dt)).
            At every interior step the live region additionally mixes toward
            the knocked-in surface with the exact probability that the path
            touches the barrier during the step yet ends on the live side
            (reflection principle under the per-step-constant GBM
            coefficients the operator itself uses). No fitted constant.
        NONE: Legacy opt-out -- the bare per-step nodal jump (the pinned
            characterization discretization before the 2026-08-18 fix).
    """

    FIRST_PASSAGE = "first_passage"
    NONE = "none"

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


class HestonAnalyticalMethod(Enum):
    """Semi-analytical Heston European pricers (second level of EngineType.ANALYTICAL)."""

    LEWIS = auto()
    GATHERAL = auto()
    WEBER = auto()

    def __str__(self):
        return self.name.title()


class HestonMCScheme(Enum):
    """Heston MC time-discretization scheme (orthogonal to MonteCarloMethod RNG)."""

    EULER = auto()       # full-truncation Euler on variance
    EULERLOG = auto()    # Euler in log-spot
    QUADEXP = auto()     # Andersen (2008) quadratic-exponential
    QUADEXP_M = auto()   # QUADEXP + Andersen §4.2 exact martingale (K0*) correction
    # Appended to preserve the numeric values of the existing enum members.
    FULL_TRUNCATION_EULER = auto()  # log-spot + plain full-truncation variance Euler

    def __str__(self):
        return self.name.title()


class SABRMCScheme(Enum):
    """SABR MC time-discretization scheme (orthogonal to MonteCarloMethod RNG)."""

    LOG_EULER = auto()   # log-Euler on the shifted forward, exact GBM on alpha
    QUADEXP = auto()     # Andersen-style conditional lognormal (exact for beta=1)

    def __str__(self):
        return self.name.title()


class ADIScheme(Enum):
    """Operator-splitting scheme for 2D ADI PDE solvers (Heston and SLV)."""

    DOUGLAS = auto()
    CRAIG_SNEYD = auto()
    MCS = auto()         # modified Craig-Sneyd

    def __str__(self):
        return self.name.replace("_", " ").title()


class LeverageCalibrationMethod(Enum):
    """How the SLV leverage surface L(S,t) is calibrated."""

    MC_BINNING = auto()              # conditional E[v|S] via nonparametric binning (v1 default)
    FORWARD_FOKKER_PLANCK = auto()   # deferred project (full.md); not implemented in v1
    UNCONDITIONAL_MEAN = auto()      # opt-in approximation: E[v] (CIR mean), NOT conditional

    def __str__(self):
        return self.name.replace("_", " ").title()
