"""The two private arithmetic backends and the factorization cache (spec 2 §5.7).

Keys: (set index, round(dt, 12), θ) per backend. θ is exact (decision 6). dt
is rounded exactly as each caller passes it (np.float64 on every march since
spec 3B stage 2 (b)): intervals of nearly equal length differ by
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
from scipy.linalg import LinAlgError, get_lapack_funcs

from quantark.util.exceptions import NumericalError

from .groups import Backend, GroupState

__all__ = ["BANDED_CACHE_MAX_ENTRIES", "FactorizationCache", "operator_matrix",
           "sparse_lu_factor", "banded_factor", "factor", "sparse_lu_advance", "banded_advance"]

# enum members as module constants: read several times per step (timing gate §9.5)
_BANDED, _SPARSE_LU = Backend.BANDED, Backend.SPARSE_LU

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
        self._stores = {_SPARSE_LU: {}, _BANDED: OrderedDict()}
        self._seeds = {_SPARSE_LU: lu_seed or {}, _BANDED: banded_seed or {}}
        # The march asks for the same (set, dt, θ) step after step: the previous
        # answer per backend, for exactly equal arguments, is the entry the full
        # lookup below would return (and for BANDED it is already the LRU's newest).
        self._last = {_SPARSE_LU: None, _BANDED: None}

    @staticmethod
    def key(set_index, dt, theta) -> tuple:
        # θ exact (spec 2 decision 6); dt rounded: ULP-different intervals share one factorization, as every banked number was computed.
        return (int(set_index), round(dt, 12), float(theta))

    def get(self, backend: Backend, set_index, dt, theta, build: Callable):
        last = self._last[backend]
        if last is not None and last[0] == set_index and last[1] == dt and last[2] == theta:
            return last[3]
        value = self._get(backend, set_index, dt, theta, build)
        if self.enabled:
            self._last[backend] = (set_index, dt, theta, value)
        return value

    def _get(self, backend: Backend, set_index, dt, theta, build: Callable):
        key = self.key(set_index, dt, theta)
        hit = self._seeds[backend].get(key)
        if hit is not None:
            return hit
        if not self.enabled:
            return build()
        store = self._stores[backend]
        if key in store:
            if backend is _BANDED:
                store.move_to_end(key)
            return store[key]
        value = build()
        store[key] = value
        if backend is _BANDED:
            while len(store) > self._banded_max:
                store.popitem(last=False)
        return value

    def clear(self, backend: Optional[Backend] = None) -> None:
        for b, store in self._stores.items():
            if backend is None or b is backend:
                store.clear()
                self._last[b] = None

    def store(self, backend: Backend) -> Mapping:
        return self._stores[backend]


def operator_matrix(l, c, u, num_x):
    # base_pde_solver.py (_build_operator_matrix), verbatim
    diagonals = [l[2:-1], c[1:-1], u[1:-2]]
    return sp.diags(
        diagonals, offsets=[-1, 0, 1], shape=(num_x - 2, num_x - 2), format="csc"
    )


def sparse_lu_factor(A, dt, theta):
    # the 0.6.0 sparse-LU build branch (base_pde_solver.py), verbatim; I is the
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
    # the 0.6.0 banded build branch (snowball_pde_solver.py), verbatim
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


_GTSV, = get_lapack_funcs(("gtsv",), (np.empty(1),))


def _tridiagonal_solve(banded, rhs):
    """``solve_banded((1, 1), banded, rhs, overwrite_b=True, check_finite=False)`` without its argument
    validation: the same LAPACK ``gtsv`` call on the same operands as scipy's ``_solve_banded``, so the
    same bits. ``banded`` and ``rhs`` are float64 arrays the backend built; ``banded`` is not overwritten.

    The wrapper's validation was a third of the step's bookkeeping (spec 3B §7.3 timing gate)."""
    if rhs.size == 0:
        return np.empty_like(rhs)
    if banded.shape[-1] == 1:                    # a 1x1 system: scipy divides by the diagonal in place
        rhs /= banded[1, 0]
        return rhs
    du2, d, du, x, info = _GTSV(banded[2, :-1], banded[1, :], banded[0, 1:], rhs, False, False, False, True)
    if info > 0:
        raise LinAlgError("singular matrix")
    if info < 0:
        raise ValueError(f"illegal value in {-info}-th argument of internal gbsv/gtsv")
    return x


def _inject(rhs, l1, u2, vn_lo, vn_up, lo, up, dt, theta):
    # base_pde_solver.py (_inject_boundary_contributions): the formula and its operand order
    rhs[0] += dt * ((1.0 - theta) * l1 * vn_lo + theta * l1 * lo)
    rhs[-1] += dt * ((1.0 - theta) * u2 * vn_up + theta * u2 * up)


def factor(backend: Backend, s, dt, theta):
    """The factorization of step set ``s`` at (dt, θ) on ``backend``, through its cache.

    The one construction site: the march's advances and the session warm-up
    (``march.warm_factorizations``) both come here, so a warmed entry is the
    march's own entry, built from the same inputs.
    """
    cache = s.factorizations
    last = cache._last[backend]
    if last is not None and last[0] == s.index and last[1] == dt and last[2] == theta:
        return last[3]                      # FactorizationCache.get's own shortcut, before the closure
    if backend is _BANDED:
        return cache.get(_BANDED, s.index, dt, theta,
                         lambda: banded_factor(s.l, s.c, s.u, dt, theta))
    return cache.get(_SPARSE_LU, s.index, dt, theta,
                     lambda: sparse_lu_factor(s.operator(), dt, theta))


def sparse_lu_advance(state: GroupState, s, dt, theta, boundaries):
    """Each column in order: rhs = M1 @ v, inject, solve (base march; snowball/phoenix LU fallback)."""
    M1, M2_lu = factor(_SPARSE_LU, s, dt, theta)
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
    banded, lower1, main1, upper1 = factor(_BANDED, s, dt, theta)
    v = state.v_next
    lo, up = boundaries
    if state.group.one_dimensional_rhs:
        # phoenix_pde_solver.py (the legacy vector-surface march, banded branch)
        vn = v[1:-1, 0]
        rhs = np.empty(len(vn))
        np.multiply(main1, vn, out=rhs)
        rhs[1:] += lower1 * vn[:-1]
        rhs[:-1] += upper1 * vn[1:]
        if v.shape[0] > 2:
            _inject(rhs, s.l[1], s.u[-2], v[0, 0], v[-1, 0], lo[0], up[0], dt, theta)
        return _tridiagonal_solve(banded, rhs)[:, None]
    # snowball_pde_solver.py (the legacy two-surface value march, two columns;
    # the statistics group: its stacked indicator columns)
    vn = v[1:-1, :]
    rhs = np.empty(vn.shape)
    if vn.shape[1] <= 2:
        # the two-surface march's own form: per column on 1-D views, scalar boundary
        # injection (elementwise the same operations, so the same bits; fewer temporaries)
        for k in range(vn.shape[1]):
            col, r = vn[:, k], rhs[:, k]
            np.multiply(main1, col, out=r)
            r[1:] += lower1 * col[:-1]
            r[:-1] += upper1 * col[1:]
            if v.shape[0] > 2:
                _inject(r, s.l[1], s.u[-2], v[0, k], v[-1, k], lo[k], up[k], dt, theta)
        return _tridiagonal_solve(banded, rhs)
    # a column that is +0.0 with +0.0 rows (a statistics column of an observation the backward march
    # has not reached) is not solved: one +0.0 template column is, and its result -- the same
    # per-column arithmetic on the same operands, signed zeros included -- fills every such column
    live = _live_columns(v, lo, up)
    if live is not None and not live.all():
        idx = np.flatnonzero(live)
        sub = np.zeros((v.shape[0], idx.size + 1))
        sub[:, :idx.size] = v[:, idx]
        sub_lo, sub_up = np.zeros(idx.size + 1), np.zeros(idx.size + 1)
        sub_lo[:idx.size], sub_up[:idx.size] = lo[idx], up[idx]
        x = _banded_columns(banded, lower1, main1, upper1, s, sub, sub_lo, sub_up, dt, theta)
        out = np.empty((x.shape[0], v.shape[1]))
        out[:, idx] = x[:, :idx.size]
        out[:, ~live] = x[:, idx.size:]
        return out
    return _banded_columns(banded, lower1, main1, upper1, s, v, lo, up, dt, theta)


def _live_columns(v, lo, up):
    """Per column: does any of its bits differ from +0.0 (its values or its two rows)? ``None`` when the
    operands are not the float64 arrays the statistics group builds (then every column is solved)."""
    if not (v.dtype == np.float64 and v.flags.c_contiguous and isinstance(lo, np.ndarray)
            and isinstance(up, np.ndarray) and lo.dtype == np.float64 and up.dtype == np.float64):
        return None
    return np.any(v.view(np.uint64), axis=0) | (lo.view(np.uint64) != 0) | (up.view(np.uint64) != 0)


def _banded_columns(banded, lower1, main1, upper1, s, v, lo, up, dt, theta):
    """The multi-column banded step: the right-hand side of every column, then one solve."""
    vn = v[1:-1, :]
    rhs = np.multiply(main1[:, None], vn)
    rhs[1:, :] += lower1[:, None] * vn[:-1, :]
    rhs[:-1, :] += upper1[:, None] * vn[1:, :]
    if v.shape[0] > 2:
        _inject(rhs, s.l[1], s.u[-2], v[0, :], v[-1, :], lo, up, dt, theta)
    return _tridiagonal_solve(banded, rhs)
