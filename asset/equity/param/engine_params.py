"""
Engine configuration parameters.
"""

from dataclasses import dataclass
from util.exceptions import ValidationError


@dataclass
class EngineParams:
    """
    Configuration parameters for pricing engines.

    Attributes:
        bump_size: Bump size for finite difference method (default: 1e-4)
        bus_days_in_year: Number of business days per year (default: 252)
    """

    bump_size: float = 1e-4
    bus_days_in_year: int = 252

    def __post_init__(self):
        """Validate parameters."""
        if self.bump_size <= 0:
            raise ValidationError(f"Bump size must be positive, got {self.bump_size}")
        if self.bump_size > 0.01:  # 1% seems too large
            raise ValidationError(f"Bump size seems too large: {self.bump_size}")
        if self.bus_days_in_year <= 0:
            raise ValidationError(
                f"Business days must be positive, got {self.bus_days_in_year}"
            )


@dataclass
class MCParams(EngineParams):
    """
    Monte Carlo engine configuration.

    Attributes:
        seed: Random seed for reproducibility
        num_paths: Number of simulation paths
        time_steps: Number of time steps per path
        use_qmc: Use quasi-Monte Carlo (default: False)
        use_antithetic: Use antithetic variates (default: False)
    """

    seed: int = 42
    num_paths: int = 10000
    time_steps: int = 100
    use_qmc: bool = False
    use_antithetic: bool = False

    def __post_init__(self):
        """Validate MC parameters."""
        super().__post_init__()
        if self.num_paths <= 0:
            raise ValidationError(
                f"Number of paths must be positive, got {self.num_paths}"
            )
        if self.time_steps <= 0:
            raise ValidationError(f"Time steps must be positive, got {self.time_steps}")


@dataclass
class PDEParams(EngineParams):
    """
    PDE engine configuration.

    Attributes:
        grid_size: Number of spatial grid points (default: 400)
        time_steps: Number of time steps (default: 200)
        adaptive_grid: Use adaptive grid spacing with Tavella-Randall (default: False)
        s_min: Lower bound for spatial grid (0 = auto-calculate)
        s_max: Upper bound for spatial grid (0 = auto-calculate)
        time_grid_type: Type of time grid - "uniform", "graded", or "event_clustered"
        grade_exponent: Exponent for graded time grid (higher = more clustering near maturity)
        theta: Finite difference scheme parameter (0.5 = Crank-Nicolson, 1.0 = Backward Euler)
        use_rannacher: Apply Rannacher smoothing for first steps (default: True)
        rannacher_steps: Number of backward Euler steps for smoothing (default: 1)
    """

    grid_size: int = 400
    time_steps: int = 200
    adaptive_grid: bool = False

    # Spatial grid configuration
    s_min: float = 0.0  # Auto-calculate if 0
    s_max: float = 0.0  # Auto-calculate if 0

    # Time grid configuration
    time_grid_type: str = "uniform"  # "uniform", "graded", "event_clustered"
    grade_exponent: float = 2.0

    # Numerical scheme configuration
    theta: float = 0.5  # 0.5 = Crank-Nicolson, 1.0 = Backward Euler
    use_rannacher: bool = True
    rannacher_steps: int = 1

    def __post_init__(self):
        """Validate PDE parameters."""
        super().__post_init__()
        if self.grid_size <= 0:
            raise ValidationError(f"Grid size must be positive, got {self.grid_size}")
        if self.time_steps <= 0:
            raise ValidationError(f"Time steps must be positive, got {self.time_steps}")
        if self.s_min < 0:
            raise ValidationError(f"s_min must be non-negative, got {self.s_min}")
        if self.s_max < 0:
            raise ValidationError(f"s_max must be non-negative, got {self.s_max}")
        if self.s_min > 0 and self.s_max > 0 and self.s_min >= self.s_max:
            raise ValidationError(
                f"s_min ({self.s_min}) must be less than s_max ({self.s_max})"
            )
        if self.time_grid_type not in ("uniform", "graded", "event_clustered"):
            raise ValidationError(
                f"time_grid_type must be 'uniform', 'graded', or 'event_clustered', "
                f"got '{self.time_grid_type}'"
            )
        if self.grade_exponent <= 0:
            raise ValidationError(
                f"grade_exponent must be positive, got {self.grade_exponent}"
            )
        if not 0.0 <= self.theta <= 1.0:
            raise ValidationError(f"theta must be in [0, 1], got {self.theta}")
        if self.rannacher_steps < 0:
            raise ValidationError(
                f"rannacher_steps must be non-negative, got {self.rannacher_steps}"
            )
