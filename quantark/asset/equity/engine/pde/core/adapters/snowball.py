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

from ..components import validate_readout
from ..groups import VALUE, Backend, ColumnGroup
from ..samples import SampleKind
from ..prepare import Operation, TerminalOutcome
from .base import FamilyAdapter, continuous_knock_in

__all__ = ["SnowballAdapter", "LocalVolAdapterMixin", "LocalVolSnowballAdapter", "Heston2DSnowballAdapter"]


class SnowballAdapter(FamilyAdapter):

    supports_statistics_group = True
    statistics_mode = "value"     # spec 3B stage 2 (a), Phoenix (b); KOResetAdapter pins "legacy" until (c)

    def damping_key(self, product) -> str:
        # decision 12: the contract alone, alive or knocked in (spec 3A §5.3)
        return "core-continuous-ki" if continuous_knock_in(product) else "core"

    # --- prepare ------------------------------------------------------------------

    def validate(self, product, env, request, lifecycle_state) -> None:
        # _price_with_solution / calculate_greeks / solve_life_surface preamble.
        # The product-token memo is cleared on every operation (§5.9): the token
        # is a pure function of the product, so a clear only costs a rebuild.
        solver = self.solver
        solver._product_token_memo.clear()
        if request.operation is Operation.EVENT_STATS:
            return   # the facade checked the type and the environment; the product after tau
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
        if op is Operation.EVENT_STATS:
            # the statistics sweep's preamble: no statistics once expired, then the
            # PDE compatibility check; a knock-out that fires at valuation is a terminal
            # outcome with the legacy readout's numbers (spec 3B §10 step 3). Under NODAL the
            # legacy readout still interpolates the post-event knock-in-ever column, so that
            # path marches until stage 2 (d).
            if tau <= 0:
                return TerminalOutcome("expired", None)
            solver._validate_product(product)
            if (self.supports_statistics_group and solver._use_cell_average_events()
                    and solver._is_knocked_out_at_valuation(product, spot, env)):
                from ..statistics.terminal import knocked_out_statistics
                return TerminalOutcome("knocked_out_at_valuation",
                                       knocked_out_statistics(solver, product, env, request))
            return None
        if op is Operation.SESSION_OUTPUTS and not request.short_circuit:
            # the spot curve's surface: the solve alone, no valuation-date short-circuit
            return None
        if tau <= 0:
            # Expired: return terminal payoff (Greeks zero)
            return self._outcome("expired", op, solver._calculate_terminal_value(product, spot, env))
        # Check if knocked out at valuation (KO payoff is fixed, so delta and gamma are zero)
        if solver._is_knocked_out_at_valuation(product, spot, env):
            return self._outcome("knocked_out_at_valuation", op, solver._get_immediate_ko_payoff(product, env))
        return None

    def _outcome(self, reason, op, value):
        if op is Operation.GREEKS:
            return TerminalOutcome(reason, {"price": value, "delta": 0.0, "gamma": 0.0})
        if op is Operation.SESSION_OUTPUTS:
            return TerminalOutcome(reason, self._session_outputs(value, None))
        return TerminalOutcome(reason, value)

    def solve_state(self, product, env, request):
        # _solve's state preamble, then the environment grid_request resolves
        # date schedules against (_stash_grid_request_env)
        solver = self.solver
        knocked_in = solver._prepare_solve_state(product, env)
        solver._stash_grid_request_env(env)
        if solver._profile_enabled:
            solver._reset_profile_stats()
        return {"knocked_in": bool(knocked_in), "reverse": solver._is_reverse,
                "ki_continuous": solver._ki_continuous, "ki_barrier": solver._ki_barrier}

    def statistics_family(self, prepared, streams):
        from ..statistics.snowball import SnowballStatistics
        return SnowballStatistics.create(prepared, streams)

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
        nodal = not solver._use_cell_average_events()     # NODAL: the inline path's masks (spec 3A §5.4, flip d)
        treatment = "nodal" if nodal else "cell_average"
        interior: dict = {}
        ko_stages: dict = {}
        ko_rows, ki_rows = [], []

        def _chain(prev, fn):
            if prev is None:
                return fn
            return lambda states, breach=None: fn(prev(states, breach), breach)

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

            def _ko(states, breach=None, _b=barrier, _c=cash, _up=breach_up, _hits_ki=ko_hits_ki, _rec=rec):
                # breach=None: the value's discounted redemption; a statistics block asks the breach
                # for its per-column target (spec 3B §5.1)
                tgt_alive = _c if breach is None else breach.ko(_rec, states["alive"], "alive")
                tgt_ki = _c if breach is None else breach.ko(_rec, states["ki"], "ki")
                if nodal:
                    # NODAL: the product's inclusive comparison at each node (spec 3A §5.4, flip d)
                    mask = solver._event_nodal_mask(spatial.s, _b, is_reverse, True)
                    alive = states["alive"].copy()
                    alive[mask] = tgt_alive if breach is None else tgt_alive[mask]
                    ki = states["ki"]
                    if _hits_ki:
                        ki = ki.copy()
                        ki[mask] = tgt_ki if breach is None else tgt_ki[mask]
                    return {"alive": alive, "ki": ki}
                if breach is None:
                    tgt_alive = tgt_ki = np.full_like(states["alive"], _c)
                return {
                    "alive": project_between(
                        spatial, _b, _up, tgt_alive, states["alive"]
                    ),
                    "ki": project_between(
                        spatial, _b, _up, tgt_ki, states["ki"]
                    )
                    if _hits_ki
                    else states["ki"],
                }

            ko_stages[step] = _chain(ko_stages.get(step), _ko)
            ko_rows.append(("ko", int(step), float(t_obs), barrier, float(cash), breach_up, bool(ko_hits_ki),
                            treatment))

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

                def _ki(states, breach=None, _b=float(barrier), _up=breach_up):   # a knock-in copies every column
                    if nodal:
                        mask = solver._event_nodal_mask(spatial.s, _b, is_reverse, False)
                        alive = states["alive"].copy()
                        alive[mask] = states["ki"][mask]
                        return {"alive": alive, "ki": states["ki"]}
                    return {
                        "alive": project_between(
                            spatial, _b, _up, states["ki"], states["alive"]
                        ),
                        "ki": states["ki"],
                    }

                interior[step] = _chain(interior.get(step), _ki)
                ki_rows.append(("ki", int(step), float(t_obs), float(barrier), breach_up, treatment))

        if ko_last:
            for step, stage in ko_stages.items():
                interior[step] = _chain(interior.get(step), stage)

        # Continuous KI: nodal coupling per step.
        continuous = None
        if product.has_ki_barrier and solver._ki_continuous:
            kib = float(solver._ki_barrier)
            s = spatial.s
            # the breached nodes, once per schedule: on the ascending grid they are a prefix (a suffix on
            # a reverse product), so the copy is a slice -- the same elements, without per-step masking
            breached = np.flatnonzero((s >= kib) if is_reverse else (s <= kib))
            whole = slice(int(breached[0]), int(breached[-1]) + 1) if breached.size else slice(0, 0)
            contiguous = breached.size == whole.stop - whole.start

            def continuous(step, states, _b=kib, _rev=is_reverse):
                if contiguous and _b == kib and _rev == is_reverse:
                    sel = whole
                else:
                    sel = (s >= _b) if _rev else (s <= _b)
                alive = states["alive"].copy()
                alive[sel] = states["ki"][sel]
                # FIRST_PASSAGE correction: restore the intra-step crossing
                # mass the nodal jump cannot see (see _apply_ki_jump).
                fp = solver._ki_fp
                if fp is not None and step > 0:
                    d = states["ki"] - alive
                    alive += fp.step_correction(step, s, _b, d)
                return {"alive": alive, "ki": states["ki"]}

        rows = ki_rows + ko_rows if ko_last else ko_rows + ki_rows    # the staging order
        if continuous is not None:
            rows.append(("continuous_ki", kib, is_reverse))
        return EventSchedule(interior=interior, continuous=continuous, evidence=tuple(rows))

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
        # The knocked-in rows first (spec 3A §5.2): under continuous KI the alive lower row is
        # documented to "transition to V1" and reads V1's row of this step. An alive-only group
        # (the split LU march) writes them too; the ki group rewrites the same values.
        if "ki" in group.surfaces or "alive" in group.surfaces:
            solver._set_boundary_conditions_v1(states["ki"], x_vec, s_vec, j, tau_remaining, product, env)
        if "alive" in group.surfaces:
            solver._set_boundary_conditions_v0(states["alive"], x_vec, s_vec, j, tau_remaining, product, env)

    def after_solve(self, prepared, group, states, j) -> None:
        pass

    after_solve.noop = True      # the march skips a hook marked no-op (an override is not marked)

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

    def readout_mode(self, prepared) -> str:
        # the valuation-date column is captured before today's events (_t0_readout)
        return "t0 capture" if self.solver._t0_has_events(prepared.product) else "final column"

    def _value_surface(self, sel):
        """The state whose final column is read (0 = not knocked in, 1 = knocked in)."""
        return "ki" if sel else "alive"

    def _t0_readout(self, prepared, result, sel):
        """(readout_vec, readout_override) for the selected surface (_compose_t0_readout)."""
        pre = result.captures.get("t0_pre_event")
        return (None if pre is None else pre[sel]), None

    def _columns(self, prepared, result):
        """(solution_vec, readout_vec, readout_override) of the surface selected at valuation."""
        sel = 1 if prepared.solve_state["knocked_in"] else 0
        solution_vec = result.states[self._value_surface(sel)][:, 0]
        readout_vec, readout_override = self._t0_readout(prepared, result, sel)
        return solution_vec, readout_vec, readout_override

    @staticmethod
    def _price(prepared, columns) -> float:
        # price from the pointwise-exact readout (the priority _price_with_solution used)
        solution_vec, readout_vec, readout_override = columns
        column = readout_vec if readout_vec is not None else solution_vec
        readout = prepared.readout
        if readout_override is not None:
            price = float(readout_override)
        else:
            price = readout.price(column, prepared.env.spot, prepared.layout)
        validate_readout(price, readout)
        return price

    def read(self, prepared, result):
        op = prepared.request.operation
        if op is Operation.EVENT_STATS:
            return self._read_statistics(prepared, result)
        solver, layout = self.solver, prepared.layout
        spot = prepared.env.spot
        readout = prepared.readout
        columns = self._columns(prepared, result)
        solution_vec, readout_vec, readout_override = columns
        column = readout_vec if readout_vec is not None else solution_vec

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
            )

        price = self._price(prepared, columns)
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
            stats = None if prepared.statistics is None else prepared.statistics.read(
                result.states, result.captures, price)
            return self._session_outputs(price, solution, stats)
        return price

    def _read_statistics(self, prepared, result):
        # the statistics sweep's pv: the caller's, else the value group's readout,
        # else (a product knocked out at valuation marches no value group) price()'s
        # immediate knock-out payoff
        if prepared.statistics is None:
            return None
        request = prepared.request
        if request.external_pv is not None:
            pv = request.external_pv
        elif request.price:
            pv = self._price(prepared, self._columns(prepared, result))
        else:   # the facade marches no value group for a product knocked out at valuation
            pv = self.solver._get_immediate_ko_payoff(prepared.product, prepared.env)
        return prepared.statistics.read(result.states, result.captures, pv)

    @staticmethod
    def _session_outputs(npv, solution, event_stats=None):
        from quantark.asset.equity.engine.pde.base_pde_solver import PDESessionOutputs
        return PDESessionOutputs(npv=float(npv), solution=solution, event_stats=event_stats,
                                 event_distribution=None)


class LocalVolAdapterMixin:
    """The local-vol routes: a local-vol row per step; the surface is the solve's."""

    sample_kind = SampleKind.NODAL
    statistics_model = "local_vol_1d"

    def scope(self, product, env):
        # The facades build the surface around the whole run (_with_surface); a
        # direct prepare builds its own for this environment.
        solver = self.solver
        if solver._active_lv_env is env and solver._active_lv_surface is not None:
            return nullcontext()
        return solver._surface_scope(env)

    def observation_maps(self, product, env, layout, tau) -> None:
        super().observation_maps(product, env, layout, tau)
        self.solver._active_s_vec = layout.spatial.s   # the S grid the local vol is sampled on

    def step_sets(self, product, env, layout, tau):
        self.solver._active_s_vec = layout.spatial.s   # a session builds them before any solve
        return super().step_sets(product, env, layout, tau)

    def samples(self, product, env, layout, tau):
        # a local-vol row per step on the solve's S grid (the route's _build_step_coefficients)
        from ..samples import sample_local_vol

        solver = self.solver
        return sample_local_vol(env, layout.time.t, layout.spatial.s, solver._active_lv_surface,
                                ref_strike=self._strike(product, env))

    def coefficient_key(self, product, env, layout, fps):
        # the surface the local vol is sampled from is a sampler input (decision 18):
        # no fingerprint for it, no key
        from ..identity import coefficient_key, component_fingerprint

        surface_fp = component_fingerprint(self.solver._active_lv_surface)
        if surface_fp is None:
            return None
        return coefficient_key(layout.fingerprint, fps, self._strike(product, env),
                               component_fingerprint(self.solver.params.operator), surface_fp)


class LocalVolSnowballAdapter(LocalVolAdapterMixin, SnowballAdapter):
    """``LocalVolSnowballPDESolver``."""


class Heston2DSnowballAdapter(SnowballAdapter):
    """The 2D solver's inherited 1D statistics march: the "2D" damping row (spec 3A §5.3)."""

    #: the 2D route's applicability rules also guard its 1D statistics march (spec 3B §5.6)
    route = "2D"

    def damping_key(self, product) -> str:
        return "2D"
