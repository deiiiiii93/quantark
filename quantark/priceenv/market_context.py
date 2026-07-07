"""Shared deterministic market context sampled on one time grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantark.param.term_sampling import (
    _validate_grid,
    discount_factors_on_grid,
    forward_carry_on_grid,
    step_vols_on_grid,
)
from quantark.util.exceptions import ValidationError


def _readonly_array(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.array(value, dtype=float, copy=True)
    if arr.shape != shape:
        raise ValidationError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} must be finite")
    arr.flags.writeable = False
    return arr


def _grid_index(t_grid: np.ndarray, value: float, name: str) -> int:
    idx = int(np.argmin(np.abs(t_grid - float(value))))
    if abs(float(t_grid[idx]) - float(value)) > 1e-12:
        nearest = float(t_grid[idx])
        raise ValidationError(
            f"{name}={value} is not a grid node; nearest grid node is {nearest}"
        )
    return idx


def discount_factor_between_grid_nodes(
    t_grid: np.ndarray,
    node_factors: np.ndarray,
    t0: float,
    t1: float,
) -> float:
    i0 = _grid_index(t_grid, t0, "t0")
    i1 = _grid_index(t_grid, t1, "t1")
    if i1 < i0:
        raise ValidationError("t1 must be greater than or equal to t0")
    return float(node_factors[i1] / node_factors[i0])


@dataclass(frozen=True)
class TermMarketContext:
    """Per-node and per-interval deterministic market data on one time grid."""

    t_grid: np.ndarray
    fwd_rates: np.ndarray
    fwd_carry: np.ndarray
    step_vols: np.ndarray
    node_dfs: np.ndarray
    step_dfs: np.ndarray
    carry_node_dfs: np.ndarray

    def __post_init__(self) -> None:
        grid = _validate_grid(self.t_grid)
        n = grid.size
        object.__setattr__(self, "t_grid", _readonly_array(grid, "t_grid", (n,)))
        object.__setattr__(
            self,
            "fwd_rates",
            _readonly_array(self.fwd_rates, "fwd_rates", (n - 1,)),
        )
        object.__setattr__(
            self,
            "fwd_carry",
            _readonly_array(self.fwd_carry, "fwd_carry", (n - 1,)),
        )
        object.__setattr__(
            self,
            "step_vols",
            _readonly_array(self.step_vols, "step_vols", (n - 1,)),
        )
        object.__setattr__(
            self, "node_dfs", _readonly_array(self.node_dfs, "node_dfs", (n,))
        )
        object.__setattr__(
            self,
            "step_dfs",
            _readonly_array(self.step_dfs, "step_dfs", (n - 1,)),
        )
        object.__setattr__(
            self,
            "carry_node_dfs",
            _readonly_array(self.carry_node_dfs, "carry_node_dfs", (n,)),
        )
        if np.any(self.node_dfs <= 0.0) or np.any(self.step_dfs <= 0.0):
            raise ValidationError("rate discount factors must be strictly positive")
        if np.any(self.carry_node_dfs <= 0.0):
            raise ValidationError("carry discount factors must be strictly positive")
        if not np.allclose(self.step_dfs, self.node_dfs[1:] / self.node_dfs[:-1]):
            raise ValidationError("step_dfs must equal node_dfs[1:] / node_dfs[:-1]")

    @classmethod
    def from_env(
        cls,
        pricing_env,
        t_grid: np.ndarray,
        ref_strike: float | None,
    ) -> "TermMarketContext":
        t = _validate_grid(t_grid)
        node_dfs = discount_factors_on_grid(pricing_env.rate_curve, t)
        fwd_rates = -np.log(node_dfs[1:] / node_dfs[:-1]) / np.diff(t)
        fwd_carry = forward_carry_on_grid(pricing_env.get_div_yield, t)
        carry_step_dfs = np.exp(-fwd_carry * np.diff(t))
        carry_node_dfs = np.concatenate(([1.0], np.cumprod(carry_step_dfs)))
        if ref_strike is None:
            step_vols = np.zeros(t.size - 1, dtype=float)
        else:
            step_vols = step_vols_on_grid(pricing_env.get_vol, ref_strike, t)
        return cls(
            t_grid=t,
            fwd_rates=fwd_rates,
            fwd_carry=fwd_carry,
            step_vols=step_vols,
            node_dfs=node_dfs,
            step_dfs=node_dfs[1:] / node_dfs[:-1],
            carry_node_dfs=carry_node_dfs,
        )

    def df_between(self, t0: float, t1: float) -> float:
        return discount_factor_between_grid_nodes(self.t_grid, self.node_dfs, t0, t1)

    def carry_df_between(self, t0: float, t1: float) -> float:
        return discount_factor_between_grid_nodes(
            self.t_grid,
            self.carry_node_dfs,
            t0,
            t1,
        )
