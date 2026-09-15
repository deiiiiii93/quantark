"""PDE routes (Snowball, Phoenix, KO-reset, Barrier, One-touch) with an explicit resolution status.

The PDE solvers take no lifecycle state: the twin carries the knocked-in flag
(``_otc_lifecycle_knocked_in``) and the service adds pending receivables. Near a
fixing the value function's diffusion layer shrinks below the grid; the route
then re-solves ONE refined clone (points raised to cover the layer, capped at
``INTRADAY_PDE_MAX_POINTS``) and reports the final status verbatim — never a
silent change of the time fill, never a status better than the grid delivered.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np

from quantark.intraday.engines.base import EnginePriceOutcome
from quantark.intraday.resolution import INTRADAY_PDE_MAX_POINTS, pde_resolution


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


def _grid_config(solver):
    from quantark.asset.equity.engine.pde.grid.config import resolve_config
    params = solver.params
    return resolve_config(params.accuracy, getattr(params, "grid", None)) if hasattr(params, "accuracy") else params.grid


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
        status = pde_resolution(ctx, dx_at_spot=numbers["dx_at_spot"], domain_log_width=numbers["domain_log_width"])
        if status.status == "unqualified" and status.required_points > numbers["points"]:
            target = min(status.required_points, INTRADAY_PDE_MAX_POINTS)
            if target > numbers["points"]:
                refined = deepcopy(engine)
                grid = _grid_config(refined)
                refined.params.grid = replace(grid, points=target, max_points=max(int(grid.max_points or 0), target))
                refined._grid_binder = None
                refined._active_layout = None
                pv = float(refined.price(twin, env))
                numbers = _layout_numbers(refined, ctx.spot)
                status = pde_resolution(ctx, dx_at_spot=numbers["dx_at_spot"], domain_log_width=numbers["domain_log_width"])
                records = (f"grid refined to {numbers['points']} points for the diffusion layer",)
                solver = refined
        numerical = dict(numbers, resolution=status.status, resolution_reason=status.reason,
                         cells_per_layer=status.cells_per_layer, layer_log_width=status.layer_log_width)
        return EnginePriceOutcome(pv, self._method(engine), numerical, {}, records, engine_used=solver)

    @staticmethod
    def _method(engine) -> str:
        params = engine.params
        rannacher = bool(getattr(params, "use_rannacher", False))
        return f"pde_{'rannacher' if rannacher else 'cn'}_theta{getattr(params, 'theta', 0.5)}"
