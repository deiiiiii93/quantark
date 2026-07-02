"""Shared backward-in-time Crank-Nicolson interior operator.

``BackwardOperator`` owns the tridiagonal INTERIOR operator (l, c, u) for a fixed
spatial grid and produces, per (dt, theta), the banded LHS system and the RHS
tridiagonal coefficients — cached, so a factorization is built once and reused
across every column solved on the same step.

Boundary conditions live in the RHS/boundary rows, NOT in this interior matrix,
so the value sweep (Dirichlet value BCs) and the indicator sweep (Neumann /
zero-slope BCs) can share ONE factorization [§11.2]. The per-step damping
schedule (``theta_by_step``) is also owned here so both sweeps run the identical
Rannacher schedule — required for price/probability consistency and for the
factorization to be shareable across sweeps.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence, Tuple

import numpy as np

from quantark.util.numerical import is_close


class BackwardOperator:
    """Fixed interior tridiagonal operator with a per-(dt, theta) banded cache."""

    def __init__(
        self,
        l: np.ndarray,
        c: np.ndarray,
        u: np.ndarray,
        *,
        use_banded: bool = True,
        cache_enabled: bool = True,
        cache_max_entries: int = 512,
    ) -> None:
        self._l = np.asarray(l, dtype=float)
        self._c = np.asarray(c, dtype=float)
        self._u = np.asarray(u, dtype=float)
        self.use_banded = use_banded
        self._cache_enabled = cache_enabled
        self._cache_max_entries = max(1, int(cache_max_entries))
        self._banded_cache: "OrderedDict[Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]" = (
            OrderedDict()
        )

    def _build(
        self, dt: float, theta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        l, c, u = self._l, self._c, self._u
        lower = -theta * dt * l[2:-1]
        main = 1.0 - theta * dt * c[1:-1]
        upper = -theta * dt * u[1:-2]

        banded = np.zeros((3, len(main)))
        banded[0, 1:] = upper
        banded[1, :] = main
        banded[2, :-1] = lower

        lower1 = (1.0 - theta) * dt * l[2:-1]
        main1 = 1.0 + (1.0 - theta) * dt * c[1:-1]
        upper1 = (1.0 - theta) * dt * u[1:-2]
        return banded, lower1, main1, upper1

    def banded_system(
        self, dt: float, theta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(banded, lower1, main1, upper1)`` for the CN step at (dt, theta).

        ``banded`` is the LHS ``I - theta*dt*A`` in scipy banded form; the three
        1-D vectors are the RHS ``I + (1-theta)*dt*A`` tridiagonals. Identical to
        the former ``SnowballPDESolver._get_banded_system`` output.
        """
        if not self._cache_enabled:
            return self._build(dt, theta)

        key = (round(dt, 12), round(theta, 12))
        cached = self._banded_cache.get(key)
        if cached is not None:
            self._banded_cache.move_to_end(key)
            return cached

        result = self._build(dt, theta)
        self._banded_cache[key] = result
        self._banded_cache.move_to_end(key)
        if len(self._banded_cache) > self._cache_max_entries:
            self._banded_cache.popitem(last=False)
        return result

    def clear_cache(self) -> None:
        self._banded_cache.clear()

    @staticmethod
    def theta_by_step(
        t_vec: np.ndarray,
        dt_vec: np.ndarray,
        params,
        discontinuity_times: Optional[Sequence[float]],
    ) -> np.ndarray:
        """Per-step Crank-Nicolson theta (length ``len(t_vec) - 1``) [§11.2].

        Reproduces the pricing sweep's Rannacher rule exactly: step ``j`` (the
        backward step from ``t_vec[j+1]`` to ``t_vec[j]``) uses backward-Euler
        (``theta = 1``) within ``rannacher_steps`` of the terminal payoff, else
        ``event_theta`` for the ``event_rannacher_steps`` steps immediately
        before each discontinuity node, else ``params.theta``.

        Depends only on ``discontinuity_times`` (always KO ∪ KI ∪ coupon), so the
        schedule is identical regardless of which streams a downstream consumer
        selects — a prerequisite for sharing the factorization across sweeps.
        """
        t_vec = np.asarray(t_vec, dtype=float)
        num_t = len(t_vec)
        n_steps = num_t - 1
        theta = np.full(n_steps, float(params.theta), dtype=float)
        if not params.use_rannacher:
            return theta

        smooth_js: set = set()
        if params.auto_grid and params.rannacher_at_events and discontinuity_times:
            for et in discontinuity_times:
                idx = int(np.argmin(np.abs(t_vec - et)))
                if 0 < idx < num_t - 1 and is_close(float(t_vec[idx]), float(et)):
                    for k in range(int(params.event_rannacher_steps)):
                        sj = idx - 1 - k
                        if sj >= 0:
                            smooth_js.add(sj)

        for j in range(n_steps):
            steps_from_end = num_t - 1 - j
            if steps_from_end < params.rannacher_steps:
                theta[j] = 1.0
            elif j in smooth_js:
                theta[j] = float(params.event_theta)
        return theta
