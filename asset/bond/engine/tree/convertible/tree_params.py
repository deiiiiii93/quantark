"""
Configuration parameters for tree-based convertible bond pricing engines.
"""
from dataclasses import dataclass
from typing import Union

from util.enum.engine_enums import ConvertibleBondTrinomialVolScheme


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
        trinomial_vol_scheme: Volatility scheme for trinomial tree engines
    """

    num_steps: int = 200
    max_iterations: int = 100
    tolerance: float = 1e-6
    bump_size: float = 0.01
    min_stock_price: float = 0.001
    max_stock_price: float = 10.0
    trinomial_vol_scheme: Union[
        str, ConvertibleBondTrinomialVolScheme
    ] = ConvertibleBondTrinomialVolScheme.CONSTANT_VOL

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
        if isinstance(self.trinomial_vol_scheme, str):
            scheme_name = self.trinomial_vol_scheme.strip().lower()
            try:
                self.trinomial_vol_scheme = ConvertibleBondTrinomialVolScheme(
                    scheme_name
                )
            except ValueError:
                try:
                    self.trinomial_vol_scheme = (
                        ConvertibleBondTrinomialVolScheme[scheme_name.upper()]
                    )
                except KeyError as exc:
                    valid = [scheme.value for scheme in ConvertibleBondTrinomialVolScheme]
                    raise ValueError(
                        f"Invalid trinomial_vol_scheme: {self.trinomial_vol_scheme}. "
                        f"Valid schemes: {valid}"
                    ) from exc
        if not isinstance(
            self.trinomial_vol_scheme, ConvertibleBondTrinomialVolScheme
        ):
            raise ValueError(
                "trinomial_vol_scheme must be a ConvertibleBondTrinomialVolScheme "
                f"or str, got {type(self.trinomial_vol_scheme)}"
            )
