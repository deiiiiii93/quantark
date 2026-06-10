"""
Configuration parameters for PDE-based convertible bond pricing engines.
"""
from dataclasses import dataclass
from util.enum.engine_enums import PDEMethod


@dataclass
class ConvertibleBondPDEParams:
    """
    Configuration parameters for PDE-based convertible bond pricing.

    Attributes:
        num_space_steps: Number of grid points in stock price dimension (default: 200)
        num_time_steps: Number of time steps (default: 500)
        scheme: PDE scheme (crank_nicolson, implicit_euler, explicit_euler)
        min_stock_multiple: Minimum stock price as multiple of current (default: 0.01)
        max_stock_multiple: Maximum stock price as multiple of current (default: 5.0)
        rannacher_steps: Number of initial Rannacher smoothing steps (default: 4)
        bump_size: Bump size for finite difference Greeks (default: 0.01)
        tolerance: Convergence tolerance for iterative schemes (default: 1e-8)
    """

    num_space_steps: int = 200
    num_time_steps: int = 500
    scheme: str = "crank_nicolson"
    min_stock_multiple: float = 0.01
    max_stock_multiple: float = 5.0
    rannacher_steps: int = 4
    bump_size: float = 0.01
    tolerance: float = 1e-8

    def __post_init__(self):
        """Validate parameters."""
        if self.num_space_steps < 10:
            raise ValueError(
                f"num_space_steps must be >= 10, got {self.num_space_steps}"
            )
        if self.num_time_steps < 10:
            raise ValueError(
                f"num_time_steps must be >= 10, got {self.num_time_steps}"
            )
        valid_schemes = ["crank_nicolson", "implicit_euler", "explicit_euler"]
        if self.scheme not in valid_schemes:
            raise ValueError(
                f"scheme must be one of {valid_schemes}, got {self.scheme}"
            )
        if self.min_stock_multiple <= 0:
            raise ValueError(
                f"min_stock_multiple must be positive, got {self.min_stock_multiple}"
            )
        if self.max_stock_multiple <= 1:
            raise ValueError(
                f"max_stock_multiple must be > 1, got {self.max_stock_multiple}"
            )
        if self.rannacher_steps < 0:
            raise ValueError(
                f"rannacher_steps must be >= 0, got {self.rannacher_steps}"
            )
