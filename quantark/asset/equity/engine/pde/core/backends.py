"""The two private arithmetic backends and the factorization cache (spec 2 §5.7).

Keys: (set index, round(dt, 12), θ) per backend. θ is exact (decision 6). dt
is rounded exactly as each caller passes it (np.float64 on the value marches,
float on the statistics march): intervals of nearly equal length differ by
ULPs and share one factorization, first build wins -- every banked number was
computed that way. LU store unbounded, banded store LRU-bounded, as today.

Every arithmetic line is copied from the legacy source its comment cites; an
operand reordered here moves a last bit somewhere in the byte book.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Mapping, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded

from quantark.util.exceptions import NumericalError

from .groups import Backend, GroupState

__all__ = ["BANDED_CACHE_MAX_ENTRIES", "FactorizationCache", "operator_matrix",
           "sparse_lu_factor", "banded_factor", "sparse_lu_advance", "banded_advance"]

#: Operational cache size (spec §4.3): an implementation detail, not a knob.
BANDED_CACHE_MAX_ENTRIES = 512


class FactorizationCache:
    """Factorizations per backend, keyed ``(set index, round(dt, 12), θ)``.

    A seed (a prepared session's pack) is consulted first; with ``enabled``
    false every other lookup builds fresh and stores nothing.
    """

    def __init__(self, *, enabled: bool = True, banded_max_entries: int = BANDED_CACHE_MAX_ENTRIES,
                 lu_seed: Optional[Mapping] = None, banded_seed: Optional[Mapping] = None):
        self.enabled = enabled
        self._banded_max = banded_max_entries
        self._stores = {Backend.SPARSE_LU: {}, Backend.BANDED: OrderedDict()}
        self._seeds = {Backend.SPARSE_LU: lu_seed or {}, Backend.BANDED: banded_seed or {}}

    @classmethod
    def sharing(cls, other: "FactorizationCache", *, lu_seed: Optional[Mapping] = None,
                banded_seed: Optional[Mapping] = None) -> "FactorizationCache":
        """A cache over ``other``'s stores (the same objects) with its own seeds."""
        cache = cls(enabled=other.enabled, banded_max_entries=other._banded_max,
                    lu_seed=lu_seed, banded_seed=banded_seed)
        cache._stores = other._stores
        return cache

    @staticmethod
    def key(set_index, dt, theta) -> tuple:
        # θ exact (spec 2 decision 6); dt rounded: ULP-different intervals share one factorization, as every banked number was computed.
        return (int(set_index), round(dt, 12), float(theta))

    def get(self, backend: Backend, set_index, dt, theta, build: Callable):
        key = self.key(set_index, dt, theta)
        hit = self._seeds[backend].get(key)
        if hit is not None:
            return hit
        if not self.enabled:
            return build()
        store = self._stores[backend]
        if key in store:
            if backend is Backend.BANDED:
                store.move_to_end(key)
            return store[key]
        value = build()
        store[key] = value
        if backend is Backend.BANDED:
            while len(store) > self._banded_max:
                store.popitem(last=False)
        return value

    def clear(self, backend: Optional[Backend] = None) -> None:
        for b, store in self._stores.items():
            if backend is None or b is backend:
                store.clear()

    def store(self, backend: Backend) -> Mapping:
        return self._stores[backend]


def operator_matrix(l, c, u, num_x):
    # base_pde_solver.py (_build_operator_matrix), verbatim
    diagonals = [l[2:-1], c[1:-1], u[1:-2]]
    return sp.diags(
        diagonals, offsets=[-1, 0, 1], shape=(num_x - 2, num_x - 2), format="csc"
    )


def sparse_lu_factor(A, dt, theta):
    # base_pde_solver.py (_get_matrices build branch), verbatim; I is the
    # interior identity every caller passed (sp.eye(num_x - 2, format="csc"))
    I = sp.eye(A.shape[0], format="csc")
    M1 = I + (1.0 - theta) * dt * A
    M2 = I - theta * dt * A
    try:
        M2_lu = spla.splu(M2)
    except Exception as e:
        raise NumericalError(f"Failed to factorize matrix: {e}")
    return M1, M2_lu


def banded_factor(l, c, u, dt, theta):
    # snowball_pde_solver.py (_get_banded_system build branch), verbatim
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


def _inject(rhs, l1, u2, vn_lo, vn_up, lo, up, dt, theta):
    # base_pde_solver.py (_inject_boundary_contributions): the formula and its operand order
    rhs[0] += dt * ((1.0 - theta) * l1 * vn_lo + theta * l1 * lo)
    rhs[-1] += dt * ((1.0 - theta) * u2 * vn_up + theta * u2 * up)


def sparse_lu_advance(state: GroupState, s, dt, theta, boundaries):
    """Each column in order: rhs = M1 @ v, inject, solve (base march; snowball/phoenix LU fallback)."""
    M1, M2_lu = s.factorizations.get(Backend.SPARSE_LU, s.index, dt, theta,
                                     lambda: sparse_lu_factor(s.operator(), dt, theta))
    v = state.v_next
    lo, up = boundaries
    out = np.empty((v.shape[0] - 2, v.shape[1]))
    for k in range(v.shape[1]):
        rhs = M1 @ v[1:-1, k]
        if v.shape[0] > 2:
            _inject(rhs, s.l[1], s.u[-2], v[0, k], v[-1, k], lo[k], up[k], dt, theta)
        out[:, k] = M2_lu.solve(rhs)
    return out


def banded_advance(state: GroupState, s, dt, theta, boundaries):
    """One solve_banded call for the whole group (decision 7)."""
    banded, lower1, main1, upper1 = s.factorizations.get(
        Backend.BANDED, s.index, dt, theta, lambda: banded_factor(s.l, s.c, s.u, dt, theta))
    v = state.v_next
    lo, up = boundaries
    if state.group.one_dimensional_rhs:
        # phoenix_pde_solver.py (_step_grids banded branch)
        vn = v[1:-1, 0]
        rhs = np.empty(len(vn))
        np.multiply(main1, vn, out=rhs)
        rhs[1:] += lower1 * vn[:-1]
        rhs[:-1] += upper1 * vn[1:]
        if v.shape[0] > 2:
            _inject(rhs, s.l[1], s.u[-2], v[0, 0], v[-1, 0], lo[0], up[0], dt, theta)
        return solve_banded((1, 1), banded, rhs, overwrite_b=True, check_finite=False)[:, None]
    # snowball_pde_solver.py (_time_stepping_two_surface: value, two columns;
    # _compute_event_stats: statistics)
    vn = v[1:-1, :]
    rhs = np.empty(vn.shape)
    rhs[:] = main1[:, None] * vn
    rhs[1:, :] += lower1[:, None] * vn[:-1, :]
    rhs[:-1, :] += upper1[:, None] * vn[1:, :]
    if v.shape[0] > 2:
        _inject(rhs, s.l[1], s.u[-2], v[0, :], v[-1, :], lo, up, dt, theta)
    return solve_banded((1, 1), banded, rhs, overwrite_b=True, check_finite=False)
