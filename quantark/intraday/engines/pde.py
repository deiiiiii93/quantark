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
from math import isfinite
from types import SimpleNamespace

import numpy as np

from quantark.intraday.engines.base import TERMINATED_POINT_GREEKS, EnginePriceOutcome, PointGreeks
from quantark.intraday.resolution import (INTRADAY_PDE_MAX_POINTS, INTRADAY_PDE_MAX_STEPS, diffusion_layer, pde_resolution,
                                          time_resolved)


def _layout_numbers(solver, spot: float):
    layout = getattr(solver, "_active_layout", None)
    if layout is None:
        return None
    x = np.asarray(layout.spatial.x, dtype=float)
    i = int(np.clip(np.searchsorted(x, np.log(spot)), 1, len(x) - 1))
    dx = float(x[i] - x[i - 1])
    return {"points": int(len(x)), "dx_at_spot": dx, "domain_log_width": float(x[-1] - x[0]),
            "requested_steps": int(layout.time.requested_steps), "actual_steps": int(layout.time.actual_steps),
            "fill_scaled": bool(layout.time.fill_scaled)}


def _barrier_placement(ctx, solver) -> str:
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
    layout, b = solver._active_layout, float(barrier)
    if b in (layout.request.hard_upper, layout.request.hard_lower):
        return ""
    s = np.exp(np.asarray(layout.spatial.x, dtype=float))
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


def _dx_min(solver) -> float:
    return float(np.min(np.diff(np.asarray(solver._active_layout.spatial.x, dtype=float))))


def _time_status(ctx, solver, layout, numbers, dx_min):
    return pde_resolution(ctx, dx_at_spot=numbers["dx_at_spot"], domain_log_width=numbers["domain_log_width"],
                          time_nodes=layout.time.t, theta=solver._theta_schedule_from_layout(layout), dx_min=dx_min)


def _time_fill_for(ctx, solver, grid, numbers, dx_min):
    """(steps_per_day, requested steps) of the smallest power-of-two multiple of the current fill whose time
    layout meets the time floors on the given space grid, or the last multiple within ``INTRADAY_PDE_MAX_STEPS``."""
    from quantark.asset.equity.engine.pde.grid.time import build_time

    layout = solver._active_layout
    spd = float(grid.steps_per_day)
    best = (spd, int(layout.time.requested_steps))
    while True:
        spd *= 2.0
        candidate = build_time(layout.request, replace(grid, steps_per_day=spd, max_steps=INTRADAY_PDE_MAX_STEPS))
        if candidate.requested_steps > INTRADAY_PDE_MAX_STEPS:
            return best
        best = (spd, int(candidate.requested_steps))
        if time_resolved(_time_status(ctx, solver, SimpleNamespace(time=candidate), numbers, dx_min)):
            return best


class PDERoute:
    def price(self, ctx, engine) -> EnginePriceOutcome:
        from quantark.asset.equity.engine.pde import EuropeanPDESolver
        from quantark.asset.equity.product.option import EuropeanVanillaOption

        num, env = ctx.numerical, ctx.pricing_env
        if num.terminated:
            return EnginePriceOutcome(0.0, "terminated", {"reason": "lifecycle state is not alive; only the ledger remains"}, {})
        twin = num.product
        if isinstance(twin, EuropeanVanillaOption):
            solver = EuropeanPDESolver(deepcopy(engine.params))
            pv = float(solver.price(twin, env))
            return EnginePriceOutcome(pv, "pde_vanilla_after_ki", _layout_numbers(solver, ctx.spot) or {}, {},
                                      engine_used=solver)
        solver = deepcopy(engine)
        solver._active_layout = None
        pv = float(solver.price(twin, env))
        numbers = _layout_numbers(solver, ctx.spot)
        records = ()
        if numbers is None:          # an at-valuation event decided the price without a grid
            return EnginePriceOutcome(pv, self._method(engine), {"resolution": "not_solved"}, {}, engine_used=solver)
        grid = solver.grid_binder.config            # the resolved config the solve used (scheme knobs included)
        dx_min = _dx_min(solver)
        status = _time_status(ctx, solver, solver._active_layout, numbers, dx_min)
        if status.status == "unqualified":
            points = int(grid.points)
            if status.required_points > numbers["points"]:
                points = max(points, min(status.required_points, INTRADAY_PDE_MAX_POINTS))
            # the time floors are judged on the refined space grid: its smallest cell shrinks with the point count
            dx_min *= numbers["points"] / max(points, numbers["points"])
            spd, requested = float(grid.steps_per_day), numbers["requested_steps"]
            space_resolvable = status.required_points <= INTRADAY_PDE_MAX_POINTS
            # a layer the point cap cannot resolve stays unqualified: more steps would buy cost, not a claim
            if space_resolvable and not time_resolved(_time_status(ctx, solver, solver._active_layout, numbers, dx_min)):
                spd, requested = _time_fill_for(ctx, solver, grid, numbers, dx_min)
            if points > numbers["points"] or spd > float(grid.steps_per_day):
                refined = deepcopy(engine)
                refined.params.grid = replace(grid, points=points, max_points=max(int(grid.max_points or 0), points),
                                              steps_per_day=spd, max_steps=max(int(grid.max_steps), requested))
                refined._grid_binder = None
                refined._active_layout = None
                pv = float(refined.price(twin, env))
                numbers = _layout_numbers(refined, ctx.spot)
                status = _time_status(ctx, refined, refined._active_layout, numbers, _dx_min(refined))
                records = (f"grid refined to {numbers['points']} points and {spd:g} steps per day for the diffusion layer",)
                solver, grid = refined, refined.grid_binder.config
        placement = _barrier_placement(ctx, solver) if status.status == "resolved" else ""
        if placement:
            status = replace(status, status="unqualified", reason=placement)
        numerical = dict(numbers, resolution=status.status, resolution_reason=status.reason,
                         cells_per_layer=status.cells_per_layer, layer_log_width=status.layer_log_width,
                         steps_per_layer=status.steps_per_layer, grid_mode_damping=status.grid_mode_damping,
                         steps_per_day=float(grid.steps_per_day))
        return EnginePriceOutcome(pv, self._method(engine), numerical, {}, records, engine_used=solver)

    def point_greeks(self, ctx, engine) -> PointGreeks:
        """The solver's own stencil on the grid the route priced on; ``ok`` only on a resolved grid."""
        if ctx.numerical.terminated:
            return TERMINATED_POINT_GREEKS
        outcome = self.price(ctx, engine)
        greeks = outcome.engine_used.calculate_greeks(ctx.numerical.product, ctx.pricing_env)
        delta, gamma = float(greeks["delta"]), float(greeks["gamma"])
        if not (isfinite(delta) and isfinite(gamma)):
            return PointGreeks(None, None, "failed", f"non-finite grid stencil (delta={delta!r}, gamma={gamma!r})", "grid_stencil")
        resolution = outcome.numerical.get("resolution")
        if resolution == "not_solved":
            # an event at the valuation instant decided the claim on the known spot: constant in a neighbourhood
            # (the query spot on that event's level is caught before the route is asked)
            if delta == 0.0 and gamma == 0.0:
                return PointGreeks(0.0, 0.0, "ok", "", "decided_at_valuation")
            return PointGreeks(None, None, "unqualified", "decided at the valuation instant without a grid", "grid_stencil")
        if resolution != "resolved":
            return PointGreeks(None, None, "unqualified", str(outcome.numerical.get("resolution_reason") or resolution),
                               "grid_stencil")
        return PointGreeks(delta, gamma, "ok", "", "grid_stencil")

    @staticmethod
    def _method(engine) -> str:
        params = engine.params
        rannacher = bool(getattr(params, "use_rannacher", False))
        return f"pde_{'rannacher' if rannacher else 'cn'}_theta{getattr(params, 'theta', 0.5)}"
