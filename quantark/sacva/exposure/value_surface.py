"""Value-surface repricing backends (spec §3.2).

The repricer depends only on the Markovian contract
``value_at(states, t, discrete_state) -> values`` — the value of the *remaining*
contract at future time ``t`` given an array of simulated ``states`` (vectorized
over paths) and the path's ``discrete_state``. The trade value at ``t`` must be a
function of ``(state, tau=T-t, discrete_state)`` only; engines that cannot be
re-evaluated at an arbitrary future state must raise upstream (no inception-only
fallback).
"""

from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError


class ValueSurface(Protocol):
    currency: str

    def value_at(self, states: np.ndarray, t: float,
                 discrete_state: Optional[str]) -> np.ndarray:
        ...


class GridValueSurface:
    """PDE/QUAD backend: interpolate a precomputed value grid per (t, state)."""

    def __init__(self, times, grids: Dict[float, Dict[Optional[str], Tuple]],
                 currency: str, allow_extrapolation: bool = False):
        self.times = np.asarray(times, dtype=float)
        self.grids = grids
        self.currency = currency
        self.allow_extrapolation = allow_extrapolation

    def value_at(self, states, t, discrete_state):
        per_t = self.grids.get(t)
        if per_t is None:  # tolerance lookup (exact float keys are brittle)
            for key in self.grids:
                if abs(key - t) <= 1e-9:
                    per_t = self.grids[key]
                    break
        if per_t is None:
            raise ValidationError(f"no value grid at t={t}")
        if discrete_state not in per_t:
            raise ValidationError(f"no surface for state {discrete_state!r} at t={t}")
        spot_grid, vals = per_t[discrete_state]
        spot_grid = np.asarray(spot_grid, dtype=float)
        vals = np.asarray(vals, dtype=float)
        if spot_grid.ndim != 1 or vals.ndim != 1 or spot_grid.shape != vals.shape:
            raise ValidationError("spot_grid and vals must be equal-length 1-D arrays")
        if not (np.all(np.isfinite(spot_grid)) and np.all(np.isfinite(vals))):
            raise ValidationError("value-surface grid must be finite")
        if np.any(np.diff(spot_grid) <= 0):
            raise ValidationError("spot_grid must be strictly increasing (np.interp)")
        states = np.asarray(states, dtype=float)
        if not self.allow_extrapolation and (
            np.any(states < spot_grid[0]) or np.any(states > spot_grid[-1])
        ):
            raise ValidationError("state outside value-surface grid (extrapolation off)")
        return np.interp(states, spot_grid, np.asarray(vals, dtype=float))


class AnalyticValueSurface:
    """Closed-form backend: re-evaluate the engine at each simulated state.

    ``as_of_env(base_env, spot, t)`` returns a pricing env rolled to time ``t``
    with the given spot, using the term-structure (deterministic) curves — not a
    flat-curve shortcut. The engine must be Markovian (re-evaluable at an
    arbitrary future spot).
    """

    def __init__(self, engine, product, base_env,
                 as_of_env: Callable[[Any, float, float], Any], currency: str):
        self.engine = engine
        self.product = product
        self.base_env = base_env
        self.as_of_env = as_of_env
        self.currency = currency

    def value_at(self, states, t, discrete_state):
        states = np.asarray(states, dtype=float)
        out = np.empty_like(states)
        for i, s in enumerate(states):
            env = self.as_of_env(self.base_env, float(s), float(t))
            out[i] = self.engine.price(self.product, env)
        if not np.all(np.isfinite(out)):
            raise ValidationError("analytic value_at produced non-finite values")
        return out
