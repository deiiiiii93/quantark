"""Phoenix (``phoenix_pde_solver.py``): a value surface per memory state, plus termination surfaces.

The legacy vector-surface march advanced every column separately, in a fixed
order: ``v0[0..K]``, then ``v1[0..K]``, then the termination surfaces ``w0``,
``w1`` for EXPIRY coupons -- each a one-column banded (or sparse-LU) solve on
the step's one factorization. Each column is a group here, in that order.
Phoenix never forced θ = 1 on zero-diffusion steps; it still does not (§8).
"""
from __future__ import annotations

import numpy as np

from quantark.util.exceptions import ValidationError

from ..groups import VALUE_NO_ZERO_DIFFUSION_OVERRIDE, Backend, ColumnGroup
from .snowball import LocalVolAdapterMixin, SnowballAdapter

__all__ = ["PhoenixAdapter", "LocalVolPhoenixAdapter"]


class PhoenixAdapter(SnowballAdapter):

    def solve_state(self, product, env, request):
        state = super().solve_state(product, env, request)
        # per-solve valuation-date readout state (_solve reset it after the preamble)
        self.solver._reset_t0_readout_state()
        return state

    def observation_maps(self, product, env, layout, tau) -> None:
        super().observation_maps(product, env, layout, tau)   # registers the coupons (single site)
        solver = self.solver
        # _solve: the memory-coupon limit, right after the grids and their maps
        num_obs = len(solver._coupon_barriers)
        if product.has_memory_coupon and num_obs > 50:
            raise ValidationError(
                f"Too many observations ({num_obs}) for Memory Phoenix PDE engine. "
                "Limit is 50 to prevent performance degradation. Use MC engine instead."
            )

    def prepare_events(self, product, env, layout, tau, request):
        # No event table: the vector surfaces take the inline path. The FIRST_PASSAGE
        # state and the termination surfaces are built in terminal(), after the
        # terminal events, where the legacy march built them.
        return None

    def _max_k(self, product) -> int:
        # Determine number of memory states to track at maturity
        return len(self.solver._coupon_barriers) if product.has_memory_coupon else 0

    def groups(self, product, layout, request) -> tuple:
        use_banded = (len(layout.spatial.x) - 2) > 2
        backend = Backend.BANDED if use_banded else Backend.SPARSE_LU
        names = [f"v0[{k}]" for k in range(self._max_k(product) + 1)]
        names += [f"v1[{k}]" for k in range(self._max_k(product) + 1)]
        if self.solver._termination_surface_active(product):
            names += ["w0", "w1"]
        return tuple(ColumnGroup(n, (n,), backend, VALUE_NO_ZERO_DIFFUSION_OVERRIDE,
                                 one_dimensional_rhs=use_banded) for n in names)

    @staticmethod
    def _lists(states):
        k = 0
        v0, v1 = [], []
        while f"v0[{k}]" in states:
            v0.append(states[f"v0[{k}]"])
            v1.append(states[f"v1[{k}]"])
            k += 1
        return v0, v1

    def allocate(self, prepared) -> dict:
        # grid_v0_list[k] is the V0 surface for k missed coupons. solver._grid_v1 is
        # NOT set: Phoenix never set it, so under continuous KI the v0 lower
        # boundary reads None and falls back to 0.0.
        layout = prepared.layout
        shape = (len(layout.spatial.x), len(layout.time.t))
        max_k = self._max_k(prepared.product)
        states = {f"v0[{k}]": np.zeros(shape) for k in range(max_k + 1)}
        states.update({f"v1[{k}]": np.zeros(shape) for k in range(max_k + 1)})
        return states

    def terminal(self, prepared, states, captures) -> None:
        # _solve: the terminal vector conditions, then the terminal coupon, KI for
        # each memory state, the terminal KO; then what the march prepared first
        solver, product, env = self.solver, prepared.product, prepared.env
        x_vec, s_vec = prepared.layout.spatial.x, prepared.layout.spatial.s
        t_vec, tau = prepared.layout.time.t, prepared.tau
        grid_v0_list, grid_v1_list = self._lists(states)
        solver._set_terminal_condition_vector(grid_v0_list, grid_v1_list, x_vec, s_vec, product, env)

        terminal_tidx = len(t_vec) - 1
        coupon_obs_idx = solver._coupon_observation_indices.get(terminal_tidx)
        if coupon_obs_idx is not None:
            solver._apply_coupon_jump_vector(
                grid_v0_list, grid_v1_list, s_vec, terminal_tidx, current_time=tau,
                product=product, pricing_env=env, obs_idx=coupon_obs_idx,
            )
        if product.has_ki_barrier:
            should_apply_ki = (
                solver._ki_continuous
                or terminal_tidx in solver._ki_observation_indices
            )
            if should_apply_ki:
                for k in range(len(grid_v0_list)):
                    solver._apply_ki_jump(grid_v0_list[k], grid_v1_list[k], s_vec, terminal_tidx, product)
        # Maturity KO (kept out of _ko_observation_indices), after the terminal coupon/KI jumps
        if solver._has_terminal_ko and solver._ko_terminal_record is not None:
            solver._apply_ko_jump_vector(
                grid_v0_list, grid_v1_list, s_vec, terminal_tidx, current_time=tau,
                product=product, pricing_env=env, ko_record=solver._ko_terminal_record,
            )

        # the vector-surface march's preamble, in its order
        solver._prepare_continuous_ki_correction(product, env, t_vec)
        term = solver._prepare_termination_surfaces(product, env, t_vec, len(x_vec))
        if term is not None:
            states["w0"], states["w1"] = term

    def boundaries(self, prepared, group, states, j) -> None:
        solver, product, env = self.solver, prepared.product, prepared.env
        name = group.surfaces[0]
        grid = states[name]
        if name in ("w0", "w1"):
            solver._set_termination_boundary(grid, j)
            return
        x_vec, s_vec = prepared.layout.spatial.x, prepared.layout.spatial.s
        tau_remaining = prepared.tau - prepared.layout.time.t[j]
        if name.startswith("v1["):
            solver._set_boundary_conditions_v1(grid, x_vec, s_vec, j, tau_remaining, product, env)
        else:
            solver._set_boundary_conditions_v0(grid, x_vec, s_vec, j, tau_remaining, product, env)

    def step_events(self, prepared, states, captures, j) -> None:
        # Apply modifications (Coupons, KO, KI). The valuation-date level-0 columns
        # are captured BEFORE their events.
        solver, product = self.solver, prepared.product
        grid_v0_list, grid_v1_list = self._lists(states)
        if j == 0 and solver._t0_has_events(product):
            solver._t0_pre_event_cols = (
                grid_v0_list[0][:, 0].copy(),
                grid_v1_list[0][:, 0].copy(),
            )
        sp = prepared.layout.spatial
        solver._apply_step_modifications_vector_surface(
            grid_v0_list, grid_v1_list, sp.x, sp.s, j, prepared.tau - prepared.layout.time.t[j],
            product, prepared.env,
        )

    def _value_surface(self, sel):
        # Result is from state 0 (no accumulated memory at valuation)
        return "v1[0]" if sel else "v0[0]"

    def _t0_readout(self, prepared, result, sel):
        # the t=0 coupon capture set the solver's readout state during the events
        return self.solver._compose_t0_readout(sel)


class LocalVolPhoenixAdapter(LocalVolAdapterMixin, PhoenixAdapter):
    """``LocalVolPhoenixPDESolver``: a local-vol row per step, no flat-exact replacement."""
