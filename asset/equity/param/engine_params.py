"""
Engine configuration parameters.
"""

from dataclasses import dataclass, field
from typing import Optional
from util.exceptions import ValidationError


@dataclass
class BumpConfig:
    """
    Configuration for factor-specific bump sizes in numerical Greeks calculation.

    Each risk factor can have its own bump size and type (relative or absolute).
    Defaults follow industry conventions from Tolerance constants.

    Attributes:
        spot_bump: Relative bump size for spot delta/gamma (default: 1%).
                   Applied as: S * (1 +/- spot_bump)
        vol_bump: Absolute bump size for vega (default: 1 vol point = 0.01).
                  Applied as: sigma +/- vol_bump
        time_bump_days: Absolute bump in days for theta (default: 1 day).
        rate_bump: Absolute bump size for rho (default: 1bp = 0.0001).
                   Applied as: r +/- rate_bump
        div_bump: Absolute bump size for dividend_rho (default: 1bp = 0.0001).
                  Applied as: q +/- div_bump

    Examples:
        >>> # Use all defaults
        >>> config = BumpConfig()
        >>>
        >>> # Custom spot bump for higher precision
        >>> config = BumpConfig(spot_bump=0.001)  # 0.1% instead of 1%
        >>>
        >>> # Aggressive bumps for illiquid securities
        >>> config = BumpConfig(vol_bump=0.02, rate_bump=0.001)  # 2 vol pts, 10bp
    """

    # Spot bump (relative) - for delta/gamma
    spot_bump: float = 0.01

    # Volatility bump (absolute) - for vega
    vol_bump: float = 0.01

    # Time bump (absolute in days) - for theta
    time_bump_days: int = 1

    # Rate bump (absolute) - for rho
    rate_bump: float = 0.0001

    # Dividend bump (absolute) - for dividend_rho
    div_bump: float = 0.0001

    def __post_init__(self):
        """Validate bump sizes."""
        # Spot bump: must be positive and reasonable (0.01% to 10%)
        if self.spot_bump <= 0:
            raise ValidationError(f"spot_bump must be positive, got {self.spot_bump}")
        if self.spot_bump > 0.1:  # 10% seems too large
            raise ValidationError(
                f"spot_bump seems too large: {self.spot_bump} (max 10%)"
            )

        # Vol bump: must be positive and reasonable
        if self.vol_bump <= 0:
            raise ValidationError(f"vol_bump must be positive, got {self.vol_bump}")
        if self.vol_bump > 0.1:  # 10 vol points seems too large
            raise ValidationError(
                f"vol_bump seems too large: {self.vol_bump} (max 10 vol points)"
            )

        # Time bump: must be positive
        if self.time_bump_days <= 0:
            raise ValidationError(
                f"time_bump_days must be positive, got {self.time_bump_days}"
            )
        if self.time_bump_days > 30:  # More than a month seems wrong
            raise ValidationError(
                f"time_bump_days seems too large: {self.time_bump_days} (max 30 days)"
            )

        # Rate bump: must be positive
        if self.rate_bump <= 0:
            raise ValidationError(f"rate_bump must be positive, got {self.rate_bump}")
        if self.rate_bump > 0.01:  # 100bp seems too large
            raise ValidationError(
                f"rate_bump seems too large: {self.rate_bump} (max 100bp)"
            )

        # Div bump: must be positive
        if self.div_bump <= 0:
            raise ValidationError(f"div_bump must be positive, got {self.div_bump}")
        if self.div_bump > 0.01:  # 100bp seems too large
            raise ValidationError(
                f"div_bump seems too large: {self.div_bump} (max 100bp)"
            )

    @classmethod
    def from_tolerance(cls) -> "BumpConfig":
        """
        Create BumpConfig from Tolerance constants.

        Returns:
            BumpConfig with values from Tolerance class.
        """
        from util.numerical.constants import Tolerance

        return cls(
            spot_bump=Tolerance.BUMP_SPOT,
            vol_bump=Tolerance.BUMP_VOL,
            rate_bump=Tolerance.BUMP_RATE,
            div_bump=Tolerance.BUMP_RATE,  # Same as rate bump
        )

    def get_bump_for_factor(self, factor: str) -> float:
        """
        Get bump size for a specific risk factor.

        Args:
            factor: Risk factor name ('spot', 'vol', 'time', 'rate', 'div')

        Returns:
            Bump size for the factor

        Raises:
            ValidationError: If factor is unknown
        """
        factor_map = {
            "spot": self.spot_bump,
            "vol": self.vol_bump,
            "time": float(self.time_bump_days),
            "rate": self.rate_bump,
            "div": self.div_bump,
        }
        if factor not in factor_map:
            raise ValidationError(f"Unknown risk factor: {factor}")
        return factor_map[factor]


@dataclass
class EngineParams:
    """
    Configuration parameters for pricing engines.

    Attributes:
        bump_size: DEPRECATED - Use bump_config instead. Bump size for finite
                   difference method (default: 1e-4). Only used for spot delta/gamma.
        bump_config: Factor-specific bump configuration. If None, creates default
                     from bump_size for backward compatibility.
        bus_days_in_year: Number of business days per year (default: 252)
    """

    bump_size: float = 1e-4
    bump_config: Optional[BumpConfig] = None
    bus_days_in_year: int = 252

    def __post_init__(self):
        """Validate parameters and create bump_config if needed."""
        # Legacy bump_size validation
        if self.bump_size <= 0:
            raise ValidationError(f"Bump size must be positive, got {self.bump_size}")
        if self.bump_size > 0.01:  # 1% seems too large
            raise ValidationError(f"Bump size seems too large: {self.bump_size}")
        if self.bus_days_in_year <= 0:
            raise ValidationError(
                f"Business days must be positive, got {self.bus_days_in_year}"
            )

        # Create bump_config from bump_size for backward compatibility
        if self.bump_config is None:
            self.bump_config = BumpConfig(spot_bump=self.bump_size)

    def get_effective_bump_config(self) -> BumpConfig:
        """
        Get the effective bump configuration.

        Returns bump_config if set, otherwise creates from legacy bump_size.

        Returns:
            Effective BumpConfig
        """
        if self.bump_config is not None:
            return self.bump_config
        # Fallback for safety (should not reach here due to __post_init__)
        return BumpConfig(spot_bump=self.bump_size)


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
        auto_grid: Enable feature-aware default grids (default: True)
        time_grid_type: Type of time grid - "uniform", "graded", "event_clustered", or "event_aligned"
        grade_exponent: Exponent for graded time grid (higher = more clustering near maturity)
        bus_days_in_year: Days per year for day-based grid heuristics (from EngineParams, default: 252)
        event_steps_per_day: Steps per day between observation events (default: 4)
        event_min_steps_per_interval: Minimum steps between consecutive events (default: 10)
        max_time_steps: Upper bound for auto-generated time steps (default: 5000)
        log_dx_target: Target log-price spacing near critical points for adaptive grids (default: 0.003)
        max_grid_size: Upper bound for auto-generated spatial grid points (default: 2000)
        include_spot_in_critical_points: Include spot as a critical point when auto_grid is enabled (default: True)
        rannacher_at_events: Apply Rannacher smoothing after event times when auto_grid is enabled (default: True)
        theta: Finite difference scheme parameter (0.5 = Crank-Nicolson, 1.0 = Backward Euler)
        use_rannacher: Apply Rannacher smoothing for first steps (default: True)
        rannacher_steps: Number of backward Euler steps for smoothing (default: 1)
    """

    grid_size: int = 400
    time_steps: int = 200
    adaptive_grid: bool = False

    # Feature-aware default grids
    auto_grid: bool = True

    # Spatial grid configuration
    s_min: float = 0.0  # Auto-calculate if 0
    s_max: float = 0.0  # Auto-calculate if 0

    # Time grid configuration
    time_grid_type: str = "uniform"  # "uniform", "graded", "event_clustered", "event_aligned"
    grade_exponent: float = 2.0

    # Auto-grid tuning parameters
    event_steps_per_day: int = 4
    event_min_steps_per_interval: int = 10
    max_time_steps: int = 5000
    log_dx_target: float = 0.003
    max_grid_size: int = 2000
    include_spot_in_critical_points: bool = True
    rannacher_at_events: bool = True

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
        if self.time_grid_type not in ("uniform", "graded", "event_clustered", "event_aligned"):
            raise ValidationError(
                f"time_grid_type must be 'uniform', 'graded', 'event_clustered', or 'event_aligned', "
                f"got '{self.time_grid_type}'"
            )
        if self.grade_exponent <= 0:
            raise ValidationError(
                f"grade_exponent must be positive, got {self.grade_exponent}"
            )
        if self.event_steps_per_day <= 0:
            raise ValidationError(
                f"event_steps_per_day must be positive, got {self.event_steps_per_day}"
            )
        if self.event_min_steps_per_interval <= 0:
            raise ValidationError(
                f"event_min_steps_per_interval must be positive, got {self.event_min_steps_per_interval}"
            )
        if self.max_time_steps <= 0:
            raise ValidationError(
                f"max_time_steps must be positive, got {self.max_time_steps}"
            )
        if self.log_dx_target <= 0:
            raise ValidationError(
                f"log_dx_target must be positive, got {self.log_dx_target}"
            )
        if self.max_grid_size <= 0:
            raise ValidationError(
                f"max_grid_size must be positive, got {self.max_grid_size}"
            )
        if not 0.0 <= self.theta <= 1.0:
            raise ValidationError(f"theta must be in [0, 1], got {self.theta}")
        if self.rannacher_steps < 0:
            raise ValidationError(
                f"rannacher_steps must be non-negative, got {self.rannacher_steps}"
            )
