"""The one backward loop (spec 2 §5.3).

The march knows no product: the adapter allocates and seeds the surfaces,
writes the Dirichlet rows, and applies what a product does after a solve and
at an event; the march advances each column group with the stepper.
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
    adapter, layout = prepared.adapter, prepared.layout
    cache = prepared.factorizations
    cache.clear()
    states, captures = adapter.allocate(prepared), {}
    adapter.terminal(prepared, states, captures)
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
            adapter.boundaries(prepared, group, states, j)
            v_next = np.column_stack([states[n][:, j + 1] for n in group.surfaces])
            v_next.setflags(write=False)
            lower = np.array([states[n][0, j] for n in group.surfaces])
            upper = np.array([states[n][-1, j] for n in group.surfaces])
            out = stepper.step(GroupState(group, v_next), coeffs.step_set(k, cache), dt, theta, (lower, upper))
            validate_advanced(out, group, n_int, stepper)
            for i, name in enumerate(group.surfaces):
                states[name][1:-1, j] = out[:, i]
            adapter.after_solve(prepared, group, states, j)
        adapter.step_events(prepared, states, captures, j)
    return MarchResult(states=states, captures=captures)
