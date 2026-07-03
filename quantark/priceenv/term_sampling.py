"""Environment-facing builder for per-interval forward market coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantark.param.term_sampling import (
    _validate_grid,
    discount_factors_on_grid,
    forward_carry_on_grid,
    forward_rates_on_grid,
    step_vols_on_grid,
)
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class TermCoefficients:
    """Per-interval forward market coefficients on a time grid.

    Interval arrays have length len(t_grid) - 1; entry i covers
    [t_grid[i], t_grid[i+1]]. Node arrays have length len(t_grid).
    Flat market inputs produce arrays where every entry equals the
    flat scalar (identity property relied on for backward compatibility).
    """

    t_grid: np.ndarray
    fwd_rates: np.ndarray
    fwd_carry: np.ndarray
    step_vols: np.ndarray
    node_dfs: np.ndarray
    step_dfs: np.ndarray

    def __post_init__(self) -> None:
        """Defensive-copy every array, freeze it, and validate shapes.

        Engines share these objects; aliased or mutated arrays would
        silently desynchronize coefficients from the grid.
        """
        grid = _validate_grid(self.t_grid)  # 1D, >=2 nodes, finite, increasing
        n = grid.size
        expected = {
            "t_grid": (n,),
            "fwd_rates": (n - 1,),
            "fwd_carry": (n - 1,),
            "step_vols": (n - 1,),
            "node_dfs": (n,),
            "step_dfs": (n - 1,),
        }
        for name, shape in expected.items():
            arr = np.array(getattr(self, name), dtype=float, copy=True)
            if arr.shape != shape:
                raise ValidationError(
                    f"{name} must have shape {shape}, got {arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValidationError(f"{name} must be finite")
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
        if np.any(self.node_dfs <= 0.0) or np.any(self.step_dfs <= 0.0):
            raise ValidationError("discount factors must be strictly positive")
        implied_step_dfs = self.node_dfs[1:] / self.node_dfs[:-1]
        if not np.allclose(self.step_dfs, implied_step_dfs, rtol=0.0, atol=1e-12):
            raise ValidationError(
                "step_dfs must equal node_dfs[1:] / node_dfs[:-1]"
            )

    @classmethod
    def from_env(
        cls, pricing_env, t_grid: np.ndarray, ref_strike: float
    ) -> "TermCoefficients":
        t = _validate_grid(t_grid)
        node_dfs = discount_factors_on_grid(pricing_env.rate_curve, t)
        return cls(
            t_grid=t,
            fwd_rates=forward_rates_on_grid(pricing_env.rate_curve, t),
            fwd_carry=forward_carry_on_grid(pricing_env.get_div_yield, t),
            step_vols=step_vols_on_grid(pricing_env.get_vol, ref_strike, t),
            node_dfs=node_dfs,
            step_dfs=node_dfs[1:] / node_dfs[:-1],
        )
