"""The one backward loop (spec 2 §5.3).

The march knows no product: the adapter allocates and seeds the surfaces,
writes the Dirichlet rows, and applies what a product does after a solve and
at an event; the march advances each column group with the stepper. The
statistics group (§5.4, Level 1) rides along as a rolling group: its states
hold one column set, carried from step to step with their boundary rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from types import MappingProxyType

from .backends import FactorizationCache, banded_advance, factor, sparse_lu_advance
from .components import CrankNicolson, validate_advanced
from .groups import Backend, GroupState

__all__ = ["MarchResult", "march", "warm_factorizations"]

_BUILTIN_STEP = CrankNicolson.step


@dataclass(eq=False)
class MarchResult:
    states: dict
    captures: dict


def march(prepared) -> MarchResult:
    """One loop over the layout's steps, last to first (spec 2 §5.3). No product branches."""
    adapter, layout, stats = prepared.adapter, prepared.layout, prepared.statistics
    cache = prepared.factorizations
    cache.clear()
    value = prepared.request.price           # False only for a statistics-only request (2D, external pv)
    states, captures = (adapter.allocate(prepared) if value else {}), {}
    if stats is not None:
        stats.allocate(states)
    if value:
        adapter.terminal(prepared, states, captures)
    if stats is not None:
        stats.terminal(states)
    coeffs, schedule, stepper = prepared.coefficients, prepared.theta_schedule, prepared.stepper
    n_int = len(layout.spatial.x) - 2
    dt_vec = layout.time.dt
    # Loop invariants, hoisted: the per-step work below is bookkeeping around one
    # solve, and on a 1,000-step march it is most of the wall time (timing gate §9.5).
    thetas = [float(x) for x in schedule]
    set_of = [coeffs.set_at(j) for j in range(len(dt_vec))]
    step_sets = {}                                              # k -> StepSet (immutable per march)
    # the built-in stepper's step is its backend's advance (CrankNicolson.step): called directly
    # (only while CrankNicolson.step is still that dispatch: a patched step is honoured)
    builtin = type(stepper) is CrankNicolson and CrankNicolson.step is _BUILTIN_STEP
    advance = {Backend.BANDED: banded_advance, Backend.SPARSE_LU: sparse_lu_advance}
    plan = [(g, bool(g.policy.zero_diffusion_theta_one), bool(g.policy.dt_as_python_float),
             tuple(g.surfaces),
             None if g.rolling else (n_int, 1 if g.one_dimensional_rhs else len(g.surfaces)),
             advance[g.backend] if builtin else stepper.step)
            for g in prepared.groups]
    after_solve = None if getattr(type(adapter).after_solve, "noop", False) else adapter.after_solve
    for j in range(len(layout.time.t) - 2, -1, -1):
        dt = dt_vec[j]
        k = set_of[j]
        s = step_sets.get(k)
        if s is None:
            s = step_sets[k] = coeffs.step_set(k, cache)
        for group, zd_theta_one, dt_float, surfaces, shape, step in plan:
            theta = thetas[j]
            if zd_theta_one and coeffs.is_zero_diffusion(k):
                theta = 1.0  # upwind advection is monotone only fully implicit
            if group.rolling:                                   # the statistics group: it supplies its rows
                v_next = np.hstack([states[n] for n in surfaces])
                lower, upper = stats.rows(j, v_next)
                n_cols = v_next.shape[1]
            else:
                adapter.boundaries(prepared, group, states, j)
                if len(surfaces) == 1:
                    # a view of column j+1: the same values column_stack would copy
                    surface = states[surfaces[0]]
                    v_next = surface[:, j + 1:j + 2]
                    lower, upper = surface[0:1, j].copy(), surface[-1:, j].copy()
                else:
                    # column_stack's values, without its per-call list handling
                    arrays = [states[n] for n in surfaces]
                    v_next = np.empty((arrays[0].shape[0], len(arrays)))
                    lower, upper = np.empty(len(arrays)), np.empty(len(arrays))
                    for c, a in enumerate(arrays):
                        v_next[:, c] = a[:, j + 1]
                        lower[c], upper[c] = a[0, j], a[-1, j]
                n_cols = None
            if not builtin:
                v_next.setflags(write=False)   # a component gets a read-only view (§5.8); the built-in never writes
            gdt = float(dt) if dt_float else dt
            out = step(GroupState(group, v_next), s, gdt, theta, (lower, upper))
            # the validator's two checks inline; it runs (and raises, naming the
            # stepper) only when one of them fails, or for a rolling group
            if shape is None or type(out) is not np.ndarray or out.shape != shape \
                    or not np.isfinite(out).all():
                validate_advanced(out, group, n_int, stepper, n_cols=n_cols)
            if group.rolling:
                stats.store(states, out, lower, upper)
            else:
                if len(surfaces) == 1:
                    surface[1:-1, j] = out[:, 0]
                else:
                    for i, name in enumerate(surfaces):
                        states[name][1:-1, j] = out[:, i]
                if after_solve is not None:
                    after_solve(prepared, group, states, j)
        if value:
            adapter.step_events(prepared, states, captures, j)
        if stats is not None:
            stats.step_events(states, captures, j)
    return MarchResult(states=states, captures=captures)


def warm_factorizations(prepared, max_entries=None) -> dict:
    """The factorizations ``march(prepared)`` would build, built ahead (spec 2 §5.10).

    Walks the march's own order -- steps last to first, the groups in order, the
    same θ (with the zero-diffusion override) and the same ``dt`` flavour -- into
    a fresh cache, and stops once ``max_entries`` entries exist; later keys fall
    through to the march, which builds them identically. Each entry is built at
    its first march-order occurrence from the march's inputs, so it is the
    march's entry bitwise. Returns ``{backend: read-only mapping}``.
    """
    cache = FactorizationCache(enabled=True)
    coeffs, schedule, layout = prepared.coefficients, prepared.theta_schedule, prepared.layout
    dt_vec = layout.time.dt
    for j in range(len(layout.time.t) - 2, -1, -1):
        if max_entries is not None and sum(len(cache.store(b)) for b in Backend) >= max_entries:
            break
        dt = dt_vec[j]
        k = coeffs.set_at(j)
        for group in prepared.groups:
            theta = float(schedule[j])
            if group.policy.zero_diffusion_theta_one and coeffs.is_zero_diffusion(k):
                theta = 1.0
            gdt = float(dt) if group.policy.dt_as_python_float else dt
            factor(group.backend, coeffs.step_set(k, cache), gdt, theta)
    return {b: MappingProxyType(dict(cache.store(b))) for b in Backend}
