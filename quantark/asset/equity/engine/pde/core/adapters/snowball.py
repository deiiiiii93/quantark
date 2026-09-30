"""The snowball family (``snowball_pde_solver.py``): two surfaces, alive and knocked-in.

Every hook is the legacy ``_solve`` / two-surface march body it
cites, in the same order. The two surfaces advance as one banded group (two
right-hand-side columns) whenever the interior system is non-trivial, else as
two sparse-LU groups -- exactly the two solve calls the legacy march made.
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from quantark.asset.equity.engine.pde.grid import EventSchedule, project_between
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import is_close

from ..backends import FactorizationCache
from ..components import validate_readout
from ..groups import VALUE, Backend, ColumnGroup
from ..prepare import Operation, TerminalOutcome
from .base import FamilyAdapter

__all__ = ["SnowballAdapter", "LocalVolSnowballAdapter"]


class SnowballAdapter(FamilyAdapter):

    # --- prepare ------------------------------------------------------------------

    def validate(self, product, env, request, lifecycle_state) -> None:
        # _price_with_solution / calculate_greeks / solve_life_surface preamble.
        # The product-token memo is cleared on every operation (§5.9): the token
        # is a pure function of the product, so a clear only costs a rebuild.
        solver = self.solver
        solver._product_token_memo.clear()
        solver._check_product_type(product)
        if env is None:
            raise ValidationError(
                f"PricingEnvironment is required for {solver._solver_name}"
            )
        solver._validate_product(product)

    def terminal_outcome(self, product, env, request, lifecycle_state):
        solver = self.solver
        op = request.operation
        spot = env.spot
        tau = product.get_maturity(env)
        if op is Operation.LIFE_SURFACE:
            # solve_life_surface's raises, in its order
            if tau <= 0:
                raise ValidationError("a life surface needs a product with time to maturity")
            if solver._is_knocked_out_at_valuation(product, spot, env):
                raise ValidationError("a life surface needs a product that is alive at valuation")
            if any(t <= 0.0 or t >= tau for t in request.extra_times):
                raise ValidationError(f"extra_times must lie strictly inside (0, {tau})")
            return None
        if op is Operation.SESSION_OUTPUTS and not request.price:
            # the spot curve's surface: the solve alone, no valuation-date short-circuit
            return None
        if tau <= 0:
            # Expired: return terminal payoff (Greeks zero)
            return self._outcome("expired", op, solver._calculate_terminal_value(product, spot, env))
        # Check if knocked out at valuation (KO payoff is fixed, so delta and gamma are zero)
        if solver._is_knocked_out_at_valuation(product, spot, env):
            return self._outcome("knocked_out", op, solver._get_immediate_ko_payoff(product, env))
        return None

    def _outcome(self, reason, op, value):
        if op is Operation.GREEKS:
            return TerminalOutcome(reason, {"price": value, "delta": 0.0, "gamma": 0.0})
        if op is Operation.SESSION_OUTPUTS:
            return TerminalOutcome(reason, self._session_outputs(value, None))
        return TerminalOutcome(reason, value)

    def solve_state(self, product, env, request):
        # _solve's state preamble, then the environment grid_request resolves
        # date schedules against (_prepare_for_request / _stash_grid_request_env)
        solver = self.solver
        knocked_in = solver._prepare_solve_state(product, env)
        solver._stash_grid_request_env(env)
        if solver._profile_enabled:
            solver._reset_profile_stats()
        return {"knocked_in": bool(knocked_in), "reverse": solver._is_reverse,
                "ki_continuous": solver._ki_continuous, "ki_barrier": solver._ki_barrier}

    def market_checks(self, product, env, tau) -> None:
        """The two-surface solve never collapsed a smile explicitly (no guard in its _solve)."""

    def grid_request(self, product, market, tau, request):
        return self.solver.grid_request(product, market, tau, extra_times=tuple(request.extra_times))

    def prepare_events(self, product, env, layout, tau, request):
        # _solve built the interior event table right after the grids; the march's
        # FIRST_PASSAGE state is built from the same forward coefficients
        schedule = self._event_schedule(product, env, layout)
        self.solver._prepare_continuous_ki_correction(product, env, layout.time.t)
        return schedule

    def factorizations(self, session) -> FactorizationCache:
        # The value march fills the solver's own stores, which it clears first, as
        # the legacy two-surface march did: the statistics sweep reads them after it
        # (a bridge until the statistics group marches on the core, Task 9).
        solver = self.solver
        return FactorizationCache.sharing(
            solver._factorizations,
            lu_seed=None if session is None else session.lu_pack,
            banded_seed=None if session is None else session.banded_pack,
        )

    def groups(self, product, layout, request) -> tuple:
        return self.value_groups(product, layout)

    def value_groups(self, product, layout) -> tuple:
        # the legacy two-surface march: banded whenever the interior system is non-trivial
        if (len(layout.spatial.x) - 2) > 2:
            return (ColumnGroup("value", ("alive", "ki"), Backend.BANDED, VALUE),)
        return (ColumnGroup("alive", ("alive",), Backend.SPARSE_LU, VALUE),
                ColumnGroup("ki", ("ki",), Backend.SPARSE_LU, VALUE))

    def _event_schedule(
        self,
        product,
        pricing_env,
        layout,
    ):
        """Interior + continuous stages for the two-surface solve.

        Terminal (t = tau) and valuation-date (t = 0) events keep their
        certified inline handling in ``_solve`` / the t0 readout; interior
        discrete events and the continuous KI coupling live here as pure
        transforms over ``{"alive": V0_col, "ki": V1_col}``.
        """
        # moved verbatim from SnowballPDESolver.event_schedule
        solver = self.solver
        spatial = layout.spatial
        tau = layout.request.tau
        is_reverse = bool(getattr(product, "is_reverse", False))
        interior: dict = {}
        ko_stages: dict = {}

        def _chain(prev, fn):
            if prev is None:
                return fn
            return lambda states: fn(prev(states))

        # KO: both surfaces projected against the discounted redemption
        # (breach_up = up-barrier XOR reverse — certified direction rule).
        for rec in solver._get_cached_ko_records(pricing_env, product):
            t_obs = rec.observation_time
            if (
                t_obs is None
                or is_close(t_obs, 0.0)
                or is_close(t_obs, tau)
                or not (0.0 < t_obs < tau)
            ):
                continue
            step = layout.time.step_at(t_obs)
            cash = solver._cashflow_value_at_time(
                pricing_env=pricing_env,
                cashflow=rec.payoff if rec.payoff is not None else 0.0,
                current_time=t_obs,
                settlement_time=rec.settlement_time,
            )
            barrier = float(rec.barrier)
            breach_up = not is_reverse
            ko_hits_ki = solver._ko_survives_ki(product)

            def _ko(states, _b=barrier, _c=cash, _up=breach_up, _hits_ki=ko_hits_ki):
                cash_col = np.full_like(states["alive"], _c)
                return {
                    "alive": project_between(
                        spatial, _b, _up, cash_col, states["alive"]
                    ),
                    "ki": project_between(
                        spatial, _b, _up, cash_col, states["ki"]
                    )
                    if _hits_ki
                    else states["ki"],
                }

            ko_stages[step] = _chain(ko_stages.get(step), _ko)

        # The knock-out wins a tie with a knock-in observed at the same instant. With the knock-out
        # staged first (certified order) that holds while it survives a knock-in: "ki" carries it
        # too, so the copy below keeps it. Under disable_ko_after_ki "ki" does not, and the copy
        # would overwrite the knock-out wherever both levels are breached, so the knock-out is
        # staged last there (``_ko_jump_goes_last``; the two stages commute on disjoint nodes).
        ko_last = solver._ko_jump_goes_last(product)
        if not ko_last:
            interior.update(ko_stages)

        # Discrete KI: alive <- ki below the (possibly per-date) barrier.
        # Applied AFTER any coincident KO (certified order: KO first).
        if (
            product.has_ki_barrier
            and not solver._ki_continuous
            and solver._ki_nodes_in_grid(product)
        ):
            ki_profile = solver._get_cached_ki_profile(pricing_env, product)
            ki_times = ki_profile["observation_times"]
            ki_barriers = ki_profile.get("barriers") or []
            for obs_idx, t_obs in enumerate(ki_times):
                if t_obs is None or not (0.0 < t_obs < tau):
                    continue
                if is_close(t_obs, 0.0) or is_close(t_obs, tau):
                    continue
                barrier = None
                if obs_idx < len(ki_barriers):
                    barrier = ki_barriers[obs_idx]
                if barrier is None:
                    barrier = solver._ki_barrier
                step = layout.time.step_at(t_obs)
                breach_up = is_reverse  # KI is a down barrier (up if reverse)

                def _ki(states, _b=float(barrier), _up=breach_up):
                    return {
                        "alive": project_between(
                            spatial, _b, _up, states["ki"], states["alive"]
                        ),
                        "ki": states["ki"],
                    }

                interior[step] = _chain(interior.get(step), _ki)

        if ko_last:
            for step, stage in ko_stages.items():
                interior[step] = _chain(interior.get(step), stage)

        # Continuous KI: nodal coupling per step.
        continuous = None
        if product.has_ki_barrier and solver._ki_continuous:
            kib = float(solver._ki_barrier)
            s = spatial.s

            def continuous(step, states, _b=kib, _rev=is_reverse):
                mask = (s >= _b) if _rev else (s <= _b)
                alive = states["alive"].copy()
                alive[mask] = states["ki"][mask]
                # FIRST_PASSAGE correction: restore the intra-step crossing
                # mass the nodal jump cannot see (see _apply_ki_jump).
                fp = solver._ki_fp
                if fp is not None and step > 0:
                    d = states["ki"] - alive
                    alive += fp.step_correction(step, s, _b, d)
                return {"alive": alive, "ki": states["ki"]}

        return EventSchedule(interior=interior, continuous=continuous)

    # --- march --------------------------------------------------------------------

    def allocate(self, prepared) -> dict:
        layout = prepared.layout
        shape = (len(layout.spatial.x), len(layout.time.t))
        alive, ki = np.zeros(shape), np.zeros(shape)
        solver = self.solver
        solver._grid_v0, solver._grid_v1 = alive, ki   # the v0 boundary and tests read them
        return {"alive": alive, "ki": ki}

    def terminal(self, prepared, states, captures) -> None:
        # _solve: the terminal conditions, then the terminal events
        solver, product, env = self.solver, prepared.product, prepared.env
        x_vec, s_vec = prepared.layout.spatial.x, prepared.layout.spatial.s
        grid_v0, grid_v1 = states["alive"], states["ki"]
        num_t = len(prepared.layout.time.t)
        solver._set_terminal_condition_v0(grid_v0, x_vec, s_vec, product, env)
        solver._set_terminal_condition_v1(grid_v1, x_vec, s_vec, product, env)

        is_terminal_ki = product.has_ki_barrier and (
            solver._ki_continuous
            or (num_t - 1) in solver._ki_observation_indices
        )
        terminal_ko = solver._has_terminal_ko and solver._ko_terminal_record is not None
        # the knock-out wins a tie with a knock-in observed with it (see _ko_jump_goes_last)
        ko_last = terminal_ko and is_terminal_ki and solver._ko_jump_goes_last(product)

        if terminal_ko and not ko_last:
            solver._apply_terminal_ko(grid_v0, grid_v1, s_vec, product, env, solver._ko_terminal_record)
        if is_terminal_ki:
            solver._apply_ki_jump(grid_v0, grid_v1, s_vec, num_t - 1, product)
        if ko_last:
            solver._apply_terminal_ko(grid_v0, grid_v1, s_vec, product, env, solver._ko_terminal_record)

        if prepared.request.life_surface:
            # Branch columns for a life-surface readout: the terminal column
            # carries the payoff, which no event follows.
            branch_v0, branch_v1 = np.empty_like(grid_v0), np.empty_like(grid_v1)
            branch_v0[:, num_t - 1] = grid_v0[:, num_t - 1]
            branch_v1[:, num_t - 1] = grid_v1[:, num_t - 1]
            captures["branch"] = (branch_v0, branch_v1)

    def boundaries(self, prepared, group, states, j) -> None:
        solver, product, env = self.solver, prepared.product, prepared.env
        x_vec, s_vec = prepared.layout.spatial.x, prepared.layout.spatial.s
        tau_remaining = prepared.tau - prepared.layout.time.t[j]
        if "alive" in group.surfaces:
            solver._set_boundary_conditions_v0(states["alive"], x_vec, s_vec, j, tau_remaining, product, env)
        if "ki" in group.surfaces:
            solver._set_boundary_conditions_v1(states["ki"], x_vec, s_vec, j, tau_remaining, product, env)

    def after_solve(self, prepared, group, states, j) -> None:
        pass

    def step_events(self, prepared, states, captures, j) -> None:
        # the legacy two-surface march, after the solves. The valuation-date column is
        # captured BEFORE its events: t=0 events are deterministic at the known
        # spot, so the readout interpolates this smooth 0+ branch and applies
        # today's transitions pointwise instead of blending across the nodal jump.
        solver, product = self.solver, prepared.product
        grid_v0, grid_v1 = states["alive"], states["ki"]
        branch = captures.get("branch")
        if branch is not None:
            branch[0][:, j] = grid_v0[:, j]
            branch[1][:, j] = grid_v1[:, j]
        if j == 0 and solver._t0_has_events(product):
            captures["t0_pre_event"] = (grid_v0[:, 0].copy(), grid_v1[:, 0].copy())
        schedule = prepared.events
        if schedule is not None and j > 0:
            # pure stage transforms on the named blocks (j == 0 keeps the
            # certified valuation-date handling below)
            block = {"alive": grid_v0[:, j], "ki": grid_v1[:, j]}
            block = schedule.apply(j, block)
            block = schedule.continuous(j, block)
            grid_v0[:, j] = block["alive"]
            grid_v1[:, j] = block["ki"]
        else:
            sp = prepared.layout.spatial
            solver._apply_step_modifications_two_surface(
                grid_v0, grid_v1, sp.x, sp.s, j, prepared.tau - prepared.layout.time.t[j],
                product, prepared.env,
            )

    # --- read ---------------------------------------------------------------------

    def _t0_readout(self, prepared, result, sel):
        """(readout_vec, readout_override) for the selected surface (_compose_t0_readout)."""
        pre = result.captures.get("t0_pre_event")
        return (None if pre is None else pre[sel]), None

    def read(self, prepared, result):
        solver, layout = self.solver, prepared.layout
        spot = prepared.env.spot
        readout = prepared.readout
        sel = 1 if prepared.solve_state["knocked_in"] else 0
        solution_vec = result.states["ki" if sel else "alive"][:, 0]
        readout_vec, readout_override = self._t0_readout(prepared, result, sel)
        column = readout_vec if readout_vec is not None else solution_vec
        op = prepared.request.operation

        if op is Operation.LIFE_SURFACE:
            from quantark.asset.equity.engine.pde.base_pde_solver import LifeSurfaceSolution

            branch = result.captures.get("branch")
            if branch is None:
                raise PricingError("the life surface needs the march's branch columns")
            branch_v0, branch_v1 = branch
            return LifeSurfaceSolution(
                t=np.array(layout.time.t, dtype=float), x=np.array(layout.spatial.x, dtype=float),
                s=np.array(layout.spatial.s, dtype=float), v0=np.array(branch_v0, dtype=float),
                v1=np.array(branch_v1, dtype=float), step_of=dict(layout.time.step_of),
                t0_readout=None if readout_vec is None else np.array(readout_vec, dtype=float),
                knocked_in_at_valuation=bool(solver._knocked_in_at_valuation),
                requested_steps=int(layout.time.requested_steps),
                actual_steps=int(layout.time.actual_steps),
                fill_scaled=bool(layout.time.fill_scaled),
            )

        # price from the pointwise-exact readout (the priority _price_with_solution used)
        if readout_override is not None:
            price = float(readout_override)
        else:
            price = readout.price(column, spot, layout)
        validate_readout(price, readout)
        if op is Operation.GREEKS:
            # delta/gamma from the smooth 0+ branch column (a cash transition at the
            # known spot is a constant shift with zero delta/gamma)
            delta, gamma = readout.delta_gamma(column, spot, layout)
            validate_readout((delta, gamma), readout)
            return {"price": price, "delta": delta, "gamma": gamma}
        if op is Operation.SESSION_OUTPUTS:
            from quantark.asset.equity.engine.pde.base_pde_solver import PDESolutionResult

            solution = PDESolutionResult(
                solution_vec=solution_vec, x_vec=layout.spatial.x, s_vec=layout.spatial.s,
                spot_log=np.log(spot), readout_vec=readout_vec, readout_override=readout_override,
            )
            return self._session_outputs(price, solution)
        return price


class LocalVolSnowballAdapter(SnowballAdapter):
    """``LocalVolSnowballPDESolver``: a local-vol row per step; the surface is the solve's."""

    def scope(self, product, env):
        # The facades build the surface around the whole run (_with_surface); a
        # direct prepare builds its own for this environment.
        solver = self.solver
        if solver._active_lv_env is env and solver._active_lv_surface is not None:
            return nullcontext()
        return solver._surface_scope(env)

    def observation_maps(self, product, env, layout, tau) -> None:
        super().observation_maps(product, env, layout, tau)
        self.solver._active_s_vec = layout.spatial.s   # the LV _build_grids override
