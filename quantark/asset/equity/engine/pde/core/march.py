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

from .components import validate_advanced
from .groups import GroupState

__all__ = ["MarchResult", "march"]


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
    for j in range(len(layout.time.t) - 2, -1, -1):
        dt = dt_vec[j]
        k = coeffs.set_at(j)
        for group in prepared.groups:
            theta = float(schedule[j])
            if group.policy.zero_diffusion_theta_one and coeffs.is_zero_diffusion(k):
                theta = 1.0  # upwind advection is monotone only fully implicit
            if group.rolling:                                   # statistics-legacy: carry the boundary rows
                v_next = np.hstack([states[n] for n in group.surfaces])
                lower, upper = v_next[0, :].copy(), v_next[-1, :].copy()
                n_cols = v_next.shape[1]
            else:
                adapter.boundaries(prepared, group, states, j)
                v_next = np.column_stack([states[n][:, j + 1] for n in group.surfaces])
                lower = np.array([states[n][0, j] for n in group.surfaces])
                upper = np.array([states[n][-1, j] for n in group.surfaces])
                n_cols = None
            v_next.setflags(write=False)
            gdt = float(dt) if group.policy.dt_as_python_float else dt
            out = stepper.step(GroupState(group, v_next), coeffs.step_set(k, cache), gdt, theta, (lower, upper))
            validate_advanced(out, group, n_int, stepper, n_cols=n_cols)
            if group.rolling:
                col = 0
                for name in group.surfaces:
                    width = states[name].shape[1]
                    new = states[name].copy()                   # legacy: v_cur = v_next.copy()
                    new[1:-1, :] = out[:, col:col + width]
                    states[name] = new
                    col += width
            else:
                for i, name in enumerate(group.surfaces):
                    states[name][1:-1, j] = out[:, i]
                adapter.after_solve(prepared, group, states, j)
        if value:
            adapter.step_events(prepared, states, captures, j)
        if stats is not None:
            stats.step_events(states, captures, j)
    return MarchResult(states=states, captures=captures)
