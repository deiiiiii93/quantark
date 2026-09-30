"""PDE routes (Snowball, Phoenix, KO-reset, Barrier, One-touch) with an explicit resolution status.

The PDE solvers take no lifecycle state: the twin carries the knocked-in flag
(``_otc_lifecycle_knocked_in``) and the service adds pending receivables. Near a
fixing the value function's diffusion layer shrinks below the grid, in space and
in time (see ``quantark.intraday.resolution``); the route then re-solves ONE
refined clone (points raised to cover the layer, capped at
``INTRADAY_PDE_MAX_POINTS``; steps per day doubled until the time floors hold on
that space grid, capped at ``INTRADAY_PDE_MAX_STEPS``, with ``max_steps`` lifted
so the fill is never scaled) and reports the final status verbatim — never a
status better than the grid delivered.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from quantark.intraday.engines.base import (TERMINATED_POINT_GREEKS, EnginePriceOutcome, PointGreeks, decided_at_valuation,
                                            decided_point_greeks, decided_price_outcome, decides_at_valuation,
                                            point_greeks_from_estimates)
from quantark.intraday.resolution import (COEFFICIENT_BYTES_PER_CELL, INTRADAY_PDE_MAX_GRID_BYTES, INTRADAY_PDE_MAX_POINTS,
                                          INTRADAY_PDE_MAX_STEPS, UNDER_RESOLVED, diffusion_layer, pde_resolution,
                                          time_resolved)


def _solved_plan(solver, twin, env):
    """The plan of the problem ``solver`` marched for ``twin`` (spec 2 §4.7), as a dict; ``None`` when nothing
    was marched on its grid (a valuation-date outcome, or a composite whose only march is another solver's)."""
    plan = solver.explain(twin, env)
    if plan.parts:
        own = [part for part in plan.parts if part.route.get("solver") == type(solver).__name__]
        plan = own[0] if own else None
    if plan is None or not plan.layout_bound:
        return None
    return plan.to_dict()


def _layout_numbers(plan, spot: float):
    if plan is None:
        return None
    x = np.asarray(plan["space"]["x"], dtype=float)
    i = int(np.clip(np.searchsorted(x, np.log(spot)), 1, len(x) - 1))
    dx = float(x[i] - x[i - 1])
    time = plan["time"]
    return {"points": int(len(x)), "dx_at_spot": dx, "domain_log_width": float(x[-1] - x[0]),
            "requested_steps": int(time["requested_steps"]), "actual_steps": int(time["actual_steps"]),
            "fill_scaled": bool(time["fill_scaled"])}


def _barrier_placement(ctx, plan) -> str:
    """Why a barrier-bearing twin's barrier is not where the contract puts it on the solved grid ("" when it is).

    A continuous knock-out barrier is a hard domain edge of the layout (exact). Otherwise the solver overwrites
    the nodes at and beyond the barrier, so the effective barrier is the first such node: first order in dx and
    alignment-dependent unless a node sits on the barrier itself.
    """
    from quantark.util.enum.option_enums import ObservationType
    from quantark.util.numerical import is_close

    twin = ctx.numerical.product
    barrier = getattr(twin, "barrier", None)
    if barrier is None or getattr(twin, "observation_type", None) not in (ObservationType.CONTINUOUS, ObservationType.DISCRETE):
        return ""
    request, b = plan["time"]["request"], float(barrier)
    if b in (request.hard_upper, request.hard_lower):
        return ""
    s = np.exp(np.asarray(plan["space"]["x"], dtype=float))
    beyond = np.nonzero(s >= b)[0] if twin.is_up_barrier else np.nonzero(s <= b)[0]
    if len(beyond) == 0:
        return ""
    node = float(s[beyond[0]] if twin.is_up_barrier else s[beyond[-1]])
    if is_close(node, b, rel_tol=1e-12, abs_tol=0.0):
        return ""
    layer = diffusion_layer(ctx)
    offset = abs(np.log(node / b))
    return (f"the barrier {b:g} enters the grid by node overwrite at S={node:.10g}, "
            f"{offset / layer if layer > 0.0 else float('inf'):.2e} layers beyond it (first order in dx)")


def bytes_per_grid_cell(solver, twin) -> int:
    """Bytes a solve of ``solver`` keeps per points x time-node cell: its stored value surfaces plus coefficients.

    Read after a solve (a memory Phoenix learns its coupon count from the schedule it resolved).
    """
    from quantark.asset.equity.engine.pde import PhoenixPDESolver, SnowballPDESolver

    if isinstance(solver, PhoenixPDESolver):
        coupons = getattr(solver, "_coupon_barriers", None)
        states = len(coupons) if coupons is not None and getattr(twin, "has_memory_coupon", False) else 0
        surfaces = 2 * (states + 1) + 2
    elif isinstance(solver, SnowballPDESolver):
        surfaces = 2
    else:
        surfaces = 1
    return 8 * surfaces + COEFFICIENT_BYTES_PER_CELL


def _dx_min(plan) -> float:
    return float(np.min(np.diff(np.asarray(plan["space"]["x"], dtype=float))))


def _time_status(ctx, time_nodes, theta, numbers, dx_min):
    return pde_resolution(ctx, dx_at_spot=numbers["dx_at_spot"], domain_log_width=numbers["domain_log_width"],
                          time_nodes=time_nodes, theta=theta, dx_min=dx_min)


def _plan_status(ctx, plan, numbers, dx_min):
    return _time_status(ctx, plan["time"]["t"], plan["scheme"]["theta_by_step"], numbers, dx_min)


def _release_solution_grids(solver) -> None:
    """Drop the full value surfaces a two-surface solver keeps after a solve (its fresh state is None).

    The route reads only prices and the layout; a later solve on the same object rebuilds them. Without this the
    unrefined solve's surfaces coexist with the refined one, and the engine handed back carries them.
    """
    for name in ("_grid_v0", "_grid_v1"):
        if isinstance(getattr(solver, name, None), np.ndarray):
            setattr(solver, name, None)


def _with_time_factor(grid, factor: float, max_steps: int):
    """``grid`` with the time basis in force scaled by ``factor`` (spec §5.6): a budget stays a budget and a
    density stays a density, so a budget-configured request is never converted and never fails the one-basis rule."""
    if grid.time_steps is not None:
        scaled = grid.with_time_steps(int(round(grid.time_steps * factor)))
    else:
        scaled = grid.with_steps_per_day(float(grid.steps_per_day) * factor)
    return replace(scaled, max_steps=max_steps)


def _basis_label(grid, factor: float) -> str:
    if grid.time_steps is not None:
        return f"{int(round(grid.time_steps * factor))} time steps"
    return f"{float(grid.steps_per_day) * factor:g} steps per day"


def _time_fill_for(ctx, solver, plan, grid, numbers, dx_min, max_steps):
    """(time factor, requested steps) of the smallest power-of-two multiple of the current fill whose time
    layout meets the time floors on the given space grid, or the last multiple within ``max_steps``."""
    from quantark.asset.equity.engine.pde.core.components import resolve_stepper
    from quantark.asset.equity.engine.pde.grid.config import DAMPING_DEFAULTS
    from quantark.asset.equity.engine.pde.grid.time import build_time

    request = plan["time"]["request"]
    stepper = resolve_stepper(solver.params, DAMPING_DEFAULTS["core"])
    factor = 1.0
    best = (factor, int(plan["time"]["requested_steps"]))
    while True:
        factor *= 2.0
        candidate = build_time(request, _with_time_factor(grid, factor, max(max_steps, 1)))
        if candidate.requested_steps > max_steps:
            return best
        best = (factor, int(candidate.requested_steps))
        # the stepper's schedule reads the event times from the request the candidate was built from
        theta = stepper.schedule(SimpleNamespace(time=candidate, request=request))
        if time_resolved(_time_status(ctx, candidate.t, theta, numbers, dx_min)):
            return best


class PDERoute:
    def price(self, ctx, engine) -> EnginePriceOutcome:
        return self._price(ctx, engine)

    def _price(self, ctx, engine) -> EnginePriceOutcome:
        from quantark.asset.equity.engine.pde import EuropeanPDESolver
        from quantark.asset.equity.product.option import EuropeanVanillaOption

        num, env = ctx.numerical, ctx.pricing_env
        if num.terminated:
            return EnginePriceOutcome(0.0, "terminated", {"reason": "lifecycle state is not alive; only the ledger remains"}, {},
                                      exact=True)
        twin = num.product
        if decides_at_valuation(twin) and decided_at_valuation(ctx):
            # nothing is left to solve: the day-level solver's zero-maturity shortcut means "already processed"
            return decided_price_outcome(ctx)
        if isinstance(twin, EuropeanVanillaOption):
            solver = EuropeanPDESolver(deepcopy(engine.params))
            pv = float(solver.price(twin, env))
            return EnginePriceOutcome(pv, "pde_vanilla_after_ki",
                                      _layout_numbers(_solved_plan(solver, twin, env), ctx.spot) or {}, {},
                                      engine_used=solver)
        solver = deepcopy(engine)
        pv = float(solver.price(twin, env))
        _release_solution_grids(solver)
        plan = _solved_plan(solver, twin, env)
        numbers = _layout_numbers(plan, ctx.spot)
        records = ()
        if numbers is None:          # an at-valuation event decided the price without a grid
            # decided on the known spot by an event at the valuation instant: no grid, nothing to discretise
            return EnginePriceOutcome(pv, self._method(engine), {"resolution": "not_solved"}, {}, engine_used=solver,
                                      exact=True)
        grid = solver.grid_binder.config            # the resolved config the solve used (scheme knobs included)
        dx_min = _dx_min(plan)
        status = _plan_status(ctx, plan, numbers, dx_min)
        budget_bound = False
        if status.status == UNDER_RESOLVED:
            max_cells = INTRADAY_PDE_MAX_GRID_BYTES // bytes_per_grid_cell(solver, twin)
            points = int(grid.points)
            if status.required_points > numbers["points"]:
                points = max(points, min(status.required_points, INTRADAY_PDE_MAX_POINTS))
            steps_now = int(numbers["requested_steps"])
            if points * (steps_now + 1) > max_cells:
                points, budget_bound = max(int(grid.points), max_cells // (steps_now + 1)), True
            # the time floors are judged on the refined space grid: its smallest cell shrinks with the point count
            dx_min *= numbers["points"] / max(points, numbers["points"])
            factor, requested = 1.0, steps_now
            space_resolvable = status.required_points <= max(points, numbers["points"])
            # a layer the point or memory cap cannot resolve stays under-resolved: more steps would buy cost, not accuracy
            if space_resolvable and not time_resolved(_plan_status(ctx, plan, numbers, dx_min)):
                max_steps = min(INTRADAY_PDE_MAX_STEPS, max_cells // points - 1)
                factor, requested = _time_fill_for(ctx, solver, plan, grid, numbers, dx_min, max_steps)
                budget_bound = budget_bound or max_steps < INTRADAY_PDE_MAX_STEPS
            if points > numbers["points"] or factor > 1.0:
                refined = deepcopy(engine)
                refined.params.grid = replace(_with_time_factor(grid, factor, max(int(grid.max_steps), requested)),
                                              points=points, max_points=max(int(grid.max_points or 0), points))
                refined._grid_binder = None
                pv = float(refined.price(twin, env))
                _release_solution_grids(refined)
                plan = _solved_plan(refined, twin, env)
                numbers = _layout_numbers(plan, ctx.spot)
                status = _plan_status(ctx, plan, numbers, _dx_min(plan))
                records = (f"grid refined to {numbers['points']} points and {_basis_label(grid, factor)} "
                           "for the diffusion layer",)
                solver, grid = refined, refined.grid_binder.config
            if status.status == UNDER_RESOLVED and budget_bound:
                status = replace(status, reason=f"{status.reason}; refinement capped by the grid memory budget "
                                                f"({INTRADAY_PDE_MAX_GRID_BYTES / 2**30:.2f} GiB = {max_cells:.2e} "
                                                "points x time nodes for this solver)")
        placement = _barrier_placement(ctx, plan) if status.status == "resolved" else ""
        if placement:
            status = replace(status, status=UNDER_RESOLVED, reason=placement)
        numerical = dict(numbers, resolution=status.status, resolution_reason=status.reason,
                         cells_per_layer=status.cells_per_layer, layer_log_width=status.layer_log_width,
                         steps_per_layer=status.steps_per_layer, grid_mode_damping=status.grid_mode_damping,
                         steps_per_day=None if grid.steps_per_day is None else float(grid.steps_per_day))
        return EnginePriceOutcome(pv, self._method(engine), numerical, {}, records, engine_used=solver)

    def point_greeks(self, ctx, engine) -> PointGreeks:
        """The solver's own stencil on the grid the route priced on.

        The mesh resolution is a diagnostic reported alongside the number: a resolved
        mesh covers the diffusion layer with enough cells and steps, which is necessary
        for a converged stencil and not sufficient; an under-resolved one says why in the
        reason. How close the stencil is to the true derivative is measured offline.
        """
        if ctx.numerical.terminated:
            return TERMINATED_POINT_GREEKS
        if decides_at_valuation(ctx.numerical.product) and decided_at_valuation(ctx):
            return decided_point_greeks(ctx)
        outcome = self.price(ctx, engine)
        greeks = outcome.engine_used.calculate_greeks(ctx.numerical.product, ctx.pricing_env)
        delta, gamma = float(greeks["delta"]), float(greeks["gamma"])
        resolution = outcome.numerical.get("resolution")
        if resolution == "not_solved":
            # an event at the valuation instant decided the claim on the known spot: constant in a neighbourhood
            # (the query spot on that event's level is caught before the route is asked)
            if delta == 0.0 and gamma == 0.0:
                return PointGreeks(0.0, 0.0, "ok", "", "decided_at_valuation")
            return point_greeks_from_estimates(delta, gamma, "grid_stencil",
                                               note="decided at the valuation instant without a grid")
        note = "" if resolution == "resolved" else str(outcome.numerical.get("resolution_reason") or resolution)
        return point_greeks_from_estimates(delta, gamma, "grid_stencil", note=note)

    @staticmethod
    def _method(engine) -> str:
        from quantark.asset.equity.engine.pde.core.components import resolve_stepper
        from quantark.asset.equity.engine.pde.grid.config import DAMPING_DEFAULTS

        params = engine.params
        if getattr(params, "stepper", None) is None:
            return "pde_cn_theta0.5"
        st = resolve_stepper(params, DAMPING_DEFAULTS["core"])
        rannacher = int(getattr(st, "terminal_damping_steps", 0) or 0) > 0
        return f"pde_{'rannacher' if rannacher else 'cn'}_theta{getattr(st, 'theta', 0.5)}"
