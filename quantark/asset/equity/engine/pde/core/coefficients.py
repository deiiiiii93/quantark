"""The stencil a march uses (spec 2 §5.8).

``StepCoefficients`` is the operator stencil per backward step, deduplicated
by unique triple; ``CoefficientsInForce`` is the choice every legacy solve
made between it and the scalar stencil.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

import numpy as np

from .backends import FactorizationCache, operator_matrix

__all__ = ["StepCoefficients", "CoefficientsInForce", "StepSet"]


class StepCoefficients(NamedTuple):
    """Per-step (l, c, u) operator coefficient sets, deduped by unique triple.

    ``lcu_sets[set_index[j]]`` is the operator for backward step ``j`` (the
    interval [t_vec[j], t_vec[j+1]]). Flat market inputs produce exactly one
    set, preserving single-operator factorization reuse; term inputs pay the
    designed per-step rebuild (spec Component 4).

    ``zero_diffusion_sets`` marks unique sets whose step vol is exactly 0.0
    (holiday intervals under a trading-clock surface): their operator is
    first-order upwind and the sweep forces theta = 1.0 on those steps
    (spec 2026-09-01 trading-clock-vol §4.5).
    """

    lcu_sets: list
    set_index: np.ndarray
    n_unique: int
    zero_diffusion_sets: frozenset = frozenset()


@dataclass(frozen=True, eq=False)
class StepSet:
    """One coefficient set as a stepper sees it: its stencil, its operator, the cache."""
    index: int
    l: np.ndarray
    c: np.ndarray
    u: np.ndarray
    operator: Callable
    factorizations: FactorizationCache


@dataclass(eq=False)
class CoefficientsInForce:
    """The stencil a march uses: the scalar stencil when one set remains (today's
    ``step_coeffs = None if sc.n_unique == 1``), else the per-step sets."""
    flat: tuple                      # (l, c, u) from scalar_stencil(r_tau, q_tau, sigma_tau)
    sets: Optional[StepCoefficients]
    num_x: int
    _operators: dict = field(default_factory=dict)

    def set_at(self, j: int) -> int:
        return 0 if self.sets is None else int(self.sets.set_index[j])

    def lcu(self, k: int) -> tuple:
        return self.flat if self.sets is None else self.sets.lcu_sets[k]

    def is_zero_diffusion(self, k: int) -> bool:
        return self.sets is not None and k in self.sets.zero_diffusion_sets

    def operator(self, k: int):
        if k not in self._operators:
            self._operators[k] = operator_matrix(*self.lcu(k), self.num_x)
        return self._operators[k]

    def step_set(self, k: int, cache: FactorizationCache) -> StepSet:
        l, c, u = self.lcu(k)
        return StepSet(k, l, c, u, lambda: self.operator(k), cache)
