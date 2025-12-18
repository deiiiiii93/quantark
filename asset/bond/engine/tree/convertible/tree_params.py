"""
Configuration parameters for tree-based convertible bond pricing engines.
"""
from dataclasses import dataclass


@dataclass
class ConvertibleBondTreeParams:
    """
    Configuration parameters for tree-based convertible bond pricing.

    Attributes:
        num_steps: Number of time steps in the tree (default: 200)
        max_iterations: Maximum iterations for convergence (default: 100)
        tolerance: Convergence tolerance for Greeks calculations (default: 1e-6)
        bump_size: Bump size for finite difference Greeks (default: 0.01 = 1%)
        min_stock_price: Minimum stock price as fraction of spot (default: 0.001)
        max_stock_price: Maximum stock price as multiple of spot (default: 10.0)
    """

    num_steps: int = 200
    max_iterations: int = 100
    tolerance: float = 1e-6
    bump_size: float = 0.01
    min_stock_price: float = 0.001
    max_stock_price: float = 10.0

    def __post_init__(self):
        """Validate parameters."""
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.bump_size <= 0:
            raise ValueError(f"bump_size must be positive, got {self.bump_size}")
        if self.min_stock_price <= 0 or self.min_stock_price >= 1:
            raise ValueError(
                f"min_stock_price must be in (0, 1), got {self.min_stock_price}"
            )
        if self.max_stock_price <= 1:
            raise ValueError(
                f"max_stock_price must be > 1, got {self.max_stock_price}"
            )
