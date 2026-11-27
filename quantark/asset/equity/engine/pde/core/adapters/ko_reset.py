"""KO-reset (``ko_reset_snowball_pde_solver.py``): V0 fires the pre-KI schedule, V1 the post-KI one.

It differs from the snowball adapter where its deleted ``_solve`` did: the
alive check, no interior event table (every step takes the inline path, whose
override dispatches the two schedules), the staged terminal of the
not-knocked-in surface, and a readout off the post-event column 0 (it never
returned a valuation-date capture).
"""
from __future__ import annotations

import numpy as np

from quantark.util.exceptions import PricingError
from quantark.util.numerical import same_instant

from .snowball import SnowballAdapter

__all__ = ["KOResetAdapter"]


class KOResetAdapter(SnowballAdapter):

    supports_statistics_group = True   # spec 3B stage 2 (c): native, the QUAD delegation is gone

    def statistics_family(self, prepared, streams):
        from ..statistics.ko_reset import KOResetStatistics
        return KOResetStatistics.create(prepared, streams)

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

        # A knocked-in note ends at schedule 2's last fixing (owner ruling D31, run 5 T7): when that falls before the
        # grid's last node, V1 stays zero above it -- no knocked-in claim lives there -- and is seeded at that index
        # during stepping (``_v1_seed_idx``), exactly as V0 is staged at schedule 1's end below.
        solver._v1_seed_idx = None
        solver._v1_seed_values = None
        solver._v1_end = None
        knocked_in = solver._knocked_in_settlement(product, env)
        if knocked_in is not None:
            solver._v1_end = (float(knocked_in.determination_time),
                              float(knocked_in.delay_df) if knocked_in.settlement_date_governs
                              else solver._structured_terminal_delay_df)
            seed = np.zeros((num_x, 1))
            solver._set_terminal_condition_v1(seed, x_vec, s_vec, product, env)
            solver._v1_seed_values = seed[:, -1].copy()
            solver._v1_seed_idx = self._knocked_in_end_index(prepared, knocked_in.determination_time)
            grid_v1[:, -1] = 0.0
        else:
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
            and not same_instant(pre_maturity, tau)
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
            solver._apply_terminal_ko_single(grid_v0, s_vec, product, env, solver._pre_ko_terminal_record,
                                            surface="alive")

        if (
            solver._has_post_terminal_ko
            and solver._post_ko_terminal_record is not None
            and not product.barrier_config.disable_ko_after_ki
        ):
            solver._apply_terminal_ko_single(grid_v1, s_vec, product, env, solver._post_ko_terminal_record,
                                            surface="ki")

        if product.has_ki_barrier and not staged_v0:
            is_terminal_ki = solver._ki_continuous
            if not is_terminal_ki:
                if (num_t - 1) in solver._ki_observation_indices:
                    is_terminal_ki = True
            if is_terminal_ki:
                solver._apply_ki_jump(grid_v0, grid_v1, s_vec, num_t - 1, product)

        if pre_ko_last:
            solver._apply_terminal_ko_single(grid_v0, s_vec, product, env, solver._pre_ko_terminal_record,
                                            surface="alive")

        if prepared.request.life_surface:
            branch_v0, branch_v1 = np.empty_like(grid_v0), np.empty_like(grid_v1)
            branch_v0[:, num_t - 1] = grid_v0[:, num_t - 1]
            branch_v1[:, num_t - 1] = grid_v1[:, num_t - 1]
            captures["branch"] = (branch_v0, branch_v1)

    @staticmethod
    def _knocked_in_end_index(prepared, post_end: float) -> int:
        """The time index of schedule 2's last fixing (a contract instant: a grid node; time zero is index 0)."""
        solver = prepared.solver
        indices = [j for j, rec in solver._post_ko_observation_indices.items()
                   if same_instant(float(rec.observation_time), post_end)]
        if not indices:
            # no schedule-2 record left in life: the knocked-in note ends with schedule 1 (get_post_maturity_time)
            indices = [j for j, rec in solver._pre_ko_observation_indices.items()
                       if same_instant(float(rec.observation_time), post_end)]
        if not indices:
            raise PricingError(f"schedule 2's end t={post_end!r} is not a node of the KO-reset grid")
        return max(indices)

    def readout_mode(self, prepared) -> str:
        return "final column"   # no t0 capture (see _t0_readout)

    def _t0_readout(self, prepared, result, sel):
        # KO-reset's result carried no readout_vec / readout_override: price and
        # Greeks read the post-event column 0 (a characterized oddity, kept)
        return None, None
