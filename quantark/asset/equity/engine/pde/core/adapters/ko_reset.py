"""KO-reset (``ko_reset_snowball_pde_solver.py``): V0 fires the pre-KI schedule, V1 the post-KI one.

It differs from the snowball adapter where its deleted ``_solve`` did: the
alive check, no interior event table (every step takes the inline path, whose
override dispatches the two schedules), the staged terminal of the
not-knocked-in surface, and a readout off the post-event column 0 (it never
returned a valuation-date capture).
"""
from __future__ import annotations

import numpy as np

from quantark.util.numerical import Tolerance, is_close

from .snowball import SnowballAdapter

__all__ = ["KOResetAdapter"]


class KOResetAdapter(SnowballAdapter):

    #: KO-reset's event statistics are the QUAD engine's (calculate_event_stats), not a PDE group.
    supports_statistics_group = False

    def solve_state(self, product, env, request):
        # A contract past its first schedule and not knocked in has matured (rule 2 of
        # KnockOutResetSnowballOption). The knock-in that keeps it alive is one carried from an
        # earlier day: a level breached today is not tested once the first schedule has ended.
        # _solve ran it after the facade's expiry / KO-at-valuation short-circuits, so it stays here.
        product.require_alive(
            env, bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        )
        return super().solve_state(product, env, request)

    def prepare_events(self, product, env, layout, tau, request):
        # certified pre/post dispatch stays inline: no event table
        self.solver._prepare_continuous_ki_correction(product, env, layout.time.t)
        return None

    def terminal(self, prepared, states, captures) -> None:
        # _solve, verbatim: V1's terminal, then V0 staged at the pre-KI maturity
        # or seeded here, then the terminal events in the certified order
        solver, product, env = self.solver, prepared.product, prepared.env
        x_vec, s_vec = prepared.layout.spatial.x, prepared.layout.spatial.s
        grid_v0, grid_v1 = states["alive"], states["ki"]
        num_x, num_t = len(x_vec), len(prepared.layout.time.t)
        tau = prepared.tau

        solver._set_terminal_condition_v1(grid_v1, x_vec, s_vec, product, env)

        # A not-yet-knocked-in contract matures on the PRE-KI schedule; only a
        # knocked-in one runs on to `tau`. Seeding V0 at the end of the grid
        # would march it back through a period it can never reach, so when the
        # two horizons differ V0 stays zero above the pre-KI maturity -- its
        # true value there -- and is seeded at that index during stepping.
        pre_maturity = float(product.get_pre_maturity_time(env))
        solver._v0_seed_idx = None
        solver._v0_seed_values = None
        staged_v0 = (
            solver._pre_ko_observation_indices
            and pre_maturity < tau
            and not is_close(pre_maturity, tau, abs_tol=Tolerance.PRECISION)
        )
        if staged_v0:
            seed = np.zeros((num_x, 1))
            solver._set_terminal_condition_v0(seed, x_vec, s_vec, product, env)
            solver._v0_seed_values = seed[:, -1].copy()
            solver._v0_seed_idx = max(solver._pre_ko_observation_indices)
        else:
            solver._set_terminal_condition_v0(grid_v0, x_vec, s_vec, product, env)

        # The first-schedule knock-out wins a tie with a knock-in observed at the same instant
        # (``_pre_ko_goes_last``; disjoint nodes otherwise).
        pre_terminal_ko = solver._has_pre_terminal_ko and solver._pre_ko_terminal_record is not None
        pre_ko_last = pre_terminal_ko and solver._pre_ko_goes_last()
        if pre_terminal_ko and not pre_ko_last:
            solver._apply_terminal_ko_single(grid_v0, s_vec, product, env, solver._pre_ko_terminal_record)

        if (
            solver._has_post_terminal_ko
            and solver._post_ko_terminal_record is not None
            and not product.barrier_config.disable_ko_after_ki
        ):
            solver._apply_terminal_ko_single(grid_v1, s_vec, product, env, solver._post_ko_terminal_record)

        if product.has_ki_barrier and not staged_v0:
            is_terminal_ki = solver._ki_continuous
            if not is_terminal_ki:
                if (num_t - 1) in solver._ki_observation_indices:
                    is_terminal_ki = True
            if is_terminal_ki:
                solver._apply_ki_jump(grid_v0, grid_v1, s_vec, num_t - 1, product)

        if pre_ko_last:
            solver._apply_terminal_ko_single(grid_v0, s_vec, product, env, solver._pre_ko_terminal_record)

        if prepared.request.life_surface:
            branch_v0, branch_v1 = np.empty_like(grid_v0), np.empty_like(grid_v1)
            branch_v0[:, num_t - 1] = grid_v0[:, num_t - 1]
            branch_v1[:, num_t - 1] = grid_v1[:, num_t - 1]
            captures["branch"] = (branch_v0, branch_v1)

    def readout_mode(self, prepared) -> str:
        return "final column"   # no t0 capture (see _t0_readout)

    def _t0_readout(self, prepared, result, sel):
        # KO-reset's result carried no readout_vec / readout_override: price and
        # Greeks read the post-event column 0 (a characterized oddity, kept)
        return None, None
