"""Phoenix (``phoenix_pde_solver.py``): a value surface per memory state, plus termination surfaces.

The surfaces are ``v0[k]`` (not knocked in) and ``v1[k]`` (knocked in) per memory state ``k``, then the
termination surfaces ``w0``, ``w1`` for EXPIRY coupons. Without memory there is one state (``k = 0``).
With memory there are two whatever the observation count (the arrears-linear fan, spec 2026-10-15 D3):
state 0 and state 0 plus one unit of arrears -- every stage is linear in the arrears, so the value of
any arrears is read off the pair (``PhoenixPDESolver._state_arrears``).

Without memory, and on the sparse-LU small-grid path, each column is its own one-column group, in the
legacy march's order ``v0[..]``, ``v1[..]``, ``w0``, ``w1`` -- each a one-column solve on the step's one
factorization. Memory on the banded backend marches every column as ONE multi-column group (D5).
θ = 1 on zero-diffusion steps, as every value group (spec 3A §5.7).
"""
from __future__ import annotations

import numpy as np

from ..groups import VALUE, Backend, ColumnGroup
from .snowball import LocalVolAdapterMixin, SnowballAdapter

__all__ = ["PhoenixAdapter", "LocalVolPhoenixAdapter", "Heston2DPhoenixAdapter"]


class PhoenixAdapter(SnowballAdapter):

    def solve_state(self, product, env, request):
        state = super().solve_state(product, env, request)
        # per-solve valuation-date readout state (_solve reset it after the preamble)
        self.solver._reset_t0_readout_state()
        # W is per solve too: the terminal coupon stage runs before terminal() rebuilds it, and must
        # discount to the terminal payment, not read the previous solve's W (a shorter note priced
        # wrong, a longer one raised IndexError on a reused solver or PDEEngine)
        self.solver._term_w0 = self.solver._term_w1 = None
        return state

    def statistics_family(self, prepared, streams):
        from ..statistics.phoenix import PhoenixStatistics
        return PhoenixStatistics.create(prepared, streams)

    def prepare_events(self, product, env, layout, tau, request):
        # No event table: the vector surfaces take the inline path. The FIRST_PASSAGE
        # state and the termination surfaces are built in terminal(), after the
        # terminal events, where the legacy march built them. The termination tables
        # are built here too, so the statistics-only path has them (spec 3B decision 14).
        self.solver._termination_tables(product, env, layout.time.t)
        return None

    def _max_k(self, product) -> int:
        """The highest memory state: 1 with memory coupons (state 0 plus one unit of arrears), else 0."""
        return 1 if product.has_memory_coupon else 0

    def groups(self, product, layout, request) -> tuple:
        use_banded = (len(layout.spatial.x) - 2) > 2
        backend = Backend.BANDED if use_banded else Backend.SPARSE_LU
        names = [f"v0[{k}]" for k in range(self._max_k(product) + 1)]
        names += [f"v1[{k}]" for k in range(self._max_k(product) + 1)]
        if self.solver._termination_surface_active(product):
            names += ["w0", "w1"]
        if use_banded and product.has_memory_coupon:
            # one multi-column banded solve per step (spec 2026-10-15 D5)
            return (ColumnGroup("value", tuple(names), backend, VALUE),)
        return tuple(ColumnGroup(n, (n,), backend, VALUE,
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
        # grid_v0_list[k] is the V0 surface of memory state k (k units of arrears more than state 0);
        # v1[k] is its knocked-in twin. solver._grid_v1 is exposed per V0 row in boundaries(), not here.
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
        if len(group.surfaces) > 1:
            self._group_boundaries(prepared, group, states, j)
            return
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
        elif solver._ki_continuous and product.has_ki_barrier:
            # Continuous KI: the knock-in-side V0 row copies the knocked-in row of the step
            # (``_boundary_rows_v0``, ``v0_{edge}_copy_ki``). The groups march v0[0..K] before
            # v1[0..K], so V1's rows of this step are set first, as SnowballAdapter.boundaries
            # does (the v1 group rewrites the same values), and its grid is exposed as
            # ``_grid_v1`` for this row only.
            knocked_in = states["v1" + name[2:]]
            solver._set_boundary_conditions_v1(knocked_in, x_vec, s_vec, j, tau_remaining, product, env)
            saved, solver._grid_v1 = solver._grid_v1, knocked_in
            try:
                solver._set_boundary_conditions_v0(grid, x_vec, s_vec, j, tau_remaining, product, env)
            finally:
                solver._grid_v1 = saved
        else:
            solver._set_boundary_conditions_v0(grid, x_vec, s_vec, j, tau_remaining, product, env)

    def _group_boundaries(self, prepared, group, states, j) -> None:
        """The Dirichlet rows of every surface of the memory group at step ``j``.

        The rows do not depend on the memory state (the per-column march wrote the same values on each),
        so state 1 copies state 0's: a unit of arrears is worth nothing on the edges. The knocked-in rows
        go first, so that under continuous KI the knock-in-side V0 row copies V1's row of this step.
        """
        solver, product, env = self.solver, prepared.product, prepared.env
        x_vec, s_vec = prepared.layout.spatial.x, prepared.layout.spatial.s
        tau_remaining = prepared.tau - prepared.layout.time.t[j]
        v0, v1 = states["v0[0]"], states["v1[0]"]
        solver._set_boundary_conditions_v1(v1, x_vec, s_vec, j, tau_remaining, product, env)
        if solver._ki_continuous and product.has_ki_barrier:
            saved, solver._grid_v1 = solver._grid_v1, v1
            try:
                solver._set_boundary_conditions_v0(v0, x_vec, s_vec, j, tau_remaining, product, env)
            finally:
                solver._grid_v1 = saved
        else:
            solver._set_boundary_conditions_v0(v0, x_vec, s_vec, j, tau_remaining, product, env)
        for k in range(1, self._max_k(product) + 1):
            for base, state in ((v0, states[f"v0[{k}]"]), (v1, states[f"v1[{k}]"])):
                state[0, j] = base[0, j]
                state[-1, j] = base[-1, j]
        if "w0" in group.surfaces:
            solver._set_termination_boundary(states["w0"], j)
            solver._set_termination_boundary(states["w1"], j)

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


class Heston2DPhoenixAdapter(PhoenixAdapter):
    """The 2D solver's inherited 1D statistics march: the "2D" damping row (spec 3A §5.3)."""

    #: the 2D route's applicability rules also guard its 1D statistics march (spec 3B §5.6)
    route = "2D"

    def damping_key(self, product) -> str:
        return "2D"
