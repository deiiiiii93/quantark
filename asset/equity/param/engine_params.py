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
            raise ValidationError(f"Business days must be positive, got {self.bus_days_in_year}")


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
            raise ValidationError(f"Number of paths must be positive, got {self.num_paths}")
        if self.time_steps <= 0:
            raise ValidationError(f"Time steps must be positive, got {self.time_steps}")


@dataclass
class PDEParams(EngineParams):
    """
    PDE engine configuration.
    
    Attributes:
        grid_size: Number of spatial grid points
        time_steps: Number of time steps
        adaptive_grid: Use adaptive grid spacing (default: False)
    """
    grid_size: int = 100
    time_steps: int = 100
    adaptive_grid: bool = False
    
    def __post_init__(self):
        """Validate PDE parameters."""
        super().__post_init__()
        if self.grid_size <= 0:
            raise ValidationError(f"Grid size must be positive, got {self.grid_size}")
        if self.time_steps <= 0:
            raise ValidationError(f"Time steps must be positive, got {self.time_steps}")

