"""Gate C harness: one cell = (product, engine, profile, horizon, spot offset, barrier) against the reference.

Statuses (never better than the evidence):
  passed        |route - ref| <= budget + 3 ref_unc at default settings AND the last two ladder levels agree within it
  unqualified   an honest miss that is not a defect: a PDE diffusion layer below resolution, discretisation error above
                budget while the refinement ladder still converges towards the reference, or MC sampling error above budget
  failed        an inconsistency to diagnose: an exact route off its closed form, a ladder not approaching the reference,
                MC bias beyond its noise
  inconclusive  no independent reference for the cell, or the reference is limited there
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from math import exp, log, sqrt
from typing import Optional

from quantark.execution.errors import CapabilityError
from quantark.intraday import resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.request import IntradayValuationRequest

from intraday.gate_c import cells as C
from intraday.reference import budgets
from intraday.reference.budgets import GATE_C_POINTS, REFERENCE_UNCERTAINTY_MULTIPLIER as K_REF
from intraday.reference.gaussian_reference import barrier_zero_carry, reference_digital, reference_snowball

SIGMA_INNER = 0.20
MC_BASE_PATHS = 2 ** 14
#: extra points per PDE level for the phase envelope (each moves every barrier's in-cell position substantially)
PHASE_SHIFTS = (1, 2, 3)
#: Memory cap of one ladder solve (points x time nodes; two-surface solvers keep 16 bytes per cell, ~1.5 GiB here).
#: Each PDE level multiplies the cells by ~4: an uncapped level 2 on a layer-refined grid needed > 12 GiB per worker.
LADDER_MAX_GRID_CELLS = 100_000_000


def pde_level_cells(level0: dict, level: int) -> float:
    """Grid cells of PDE ladder ``level`` built from the level-0 solve (points and steps both doubled per level)."""
    return float(level0.get("points") or 0) * 2 ** level * (float(level0.get("requested_steps") or 0) * 2 ** level + 1)


@dataclass(frozen=True)
class CellResult:
    cell: C.Cell
    route_price: Optional[float]
    ref_price: Optional[float]
    ref_uncertainty: Optional[float]
    budget: float
    passed: bool
    status: str
    reason: str = ""
    numerical: dict = field(default_factory=dict)
    refinement: dict = field(default_factory=dict)


def build_context(cell: C.Cell):
    from intraday.conftest import flat_env
    fixing_ts, fixings = C.fixing_and_history(cell.product)
    ts = fixing_ts - cell.horizon
    prod, prof, (r, q) = C.product(cell.product), C.profile(cell.profile), C.market(cell.product)
    barrier = C.barrier_level(cell)
    probe = resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(ts, spot=barrier, r=r, q=q),
                                                     session_calendar=C.sse(), variance_profile=prof, fixings=fixings))
    tau = probe.numerical.maturity_tau if cell.product != "snowball_discrete_ki" else min(
        t for t in probe.numerical.event_taus.values() if t > 0.0)
    sw = sqrt(max(float(probe.pricing_env.vol_surface.total_variance(100.0, tau, barrier)), 0.0))
    kind, _, amount = cell.offset.partition("+") if "+" in cell.offset else cell.offset.partition("-")
    sign = 1.0 if "+" in cell.offset else -1.0
    if cell.offset == "eq":
        spot = barrier
    elif kind == "bp":
        spot = barrier * (1.0 + sign * float(amount) * 1e-4)
    else:
        spot = barrier * exp(sign * float(amount) * sw)
    ctx = resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(ts, spot=spot, r=r, q=q),
                                                   session_calendar=C.sse(), variance_profile=prof, fixings=fixings))
    return ctx, sw


def engine_for(cell: C.Cell, level: int = 0, level0: Optional[dict] = None, phase_shift: int = 0):
    """The engine for one ladder level. PDE levels >= 1 double the grid the route ACTUALLY solved at level 0
    (``level0`` = its numerical record): the route raises points and steps per day to the diffusion-layer floor,
    and a ladder scaling the default request instead would leave the solved grid pinned at that floor.
    ``phase_shift`` adds points at the same level: a few more points move every barrier's position inside its cell."""
    from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine, DigitalOptionAnalyticalEngine, OneTouchAnalyticalEngine
    from quantark.asset.equity.engine.mc import BarrierOptionMCEngine, DigitalOptionMCEngine, SnowballMCEngine
    from quantark.asset.equity.engine.pde import BarrierPDESolver, OneTouchPDESolver, SnowballPDESolver
    from quantark.asset.equity.engine.pde.grid.config import GridConfig, resolve_config
    from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
    from quantark.asset.equity.param import MCParams, PDEParams
    from quantark.asset.equity.param.quad_v2_params import QuadV2Params
    from quantark.util.enum.engine_enums import MonteCarloMethod
    if cell.engine == "quad_v2":
        return SnowballQuadEngineV2(QuadV2Params(cells_per_sd=2.0 * 2 ** level))
    if cell.engine == "pde":
        base = resolve_config("standard", None)
        if level == 0 and phase_shift == 0:
            params = PDEParams()
        else:
            solved = level0 or {}
            points = int(solved.get("points") or base.points) * 2 ** level + phase_shift
            spd = float(solved.get("steps_per_day") or base.steps_per_day) * 2 ** level
            params = PDEParams(grid=GridConfig(points=points, steps_per_day=spd, max_points=max(points, int(base.max_points)),
                                               max_steps=10 ** 8))
        return {"snowball_discrete_ki": SnowballPDESolver, "barrier_uo_zero_carry": BarrierPDESolver,
                "one_touch_zero_carry": OneTouchPDESolver}[cell.product](params)
    if cell.engine == "mc_rqmc":
        params = MCParams(num_paths=MC_BASE_PATHS * 4 ** level, seed=11)
        cls = {"snowball_discrete_ki": SnowballMCEngine, "barrier_uo_zero_carry": BarrierOptionMCEngine,
               "digital": DigitalOptionMCEngine}[cell.product]
        extra = {"use_brownian_bridge": True} if cls is BarrierOptionMCEngine else {}
        return cls(params=params, method=MonteCarloMethod.RANDOMIZED_QUASI, **extra)
    if cell.engine == "analytical":
        return {"digital": DigitalOptionAnalyticalEngine, "barrier_uo_zero_carry": BarrierAnalyticalEngine,
                "one_touch_zero_carry": OneTouchAnalyticalEngine}[cell.product]()
    raise ValueError(cell.engine)


_REFERENCE_CACHE: dict = {}


def reference_for(cell: C.Cell, ctx):
    """(price, uncertainty) of the remaining claim, or None when no independent reference covers the cell.

    Identical for every engine on the same (product, profile, horizon, offset, barrier): cached per process.
    """
    key = (cell.product, cell.profile, cell.horizon, cell.offset, cell.barrier)
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = _reference(cell, ctx)
    return _REFERENCE_CACHE[key]


def _reference(cell: C.Cell, ctx):
    spot = float(ctx.pricing_env.spot)
    if cell.product == "snowball_discrete_ki":
        if ctx.numerical.terminated:
            return 0.0, 0.0
        ref = reference_snowball(ctx, points=GATE_C_POINTS)
        return ref.price, ref.uncertainty_price
    if cell.product == "digital":
        return reference_digital(ctx).price, 0.0
    T = ctx.numerical.maturity_tau
    W = float(ctx.pricing_env.vol_surface.total_variance(100.0, T, spot)) if T > 0.0 else 0.0
    if cell.product == "barrier_uo_zero_carry":
        if spot >= 103.0:
            return 0.0, 0.0
        if W == 0.0:
            return max(spot - 100.0, 0.0), 0.0
        return barrier_zero_carry(spot, 100.0, 103.0, W / SIGMA_INNER ** 2, SIGMA_INNER, is_call=True, is_up=True,
                                  is_knock_out=True), 0.0
    if cell.product == "one_touch_zero_carry":
        if spot >= 103.0:
            return 1.0, 0.0
        if W == 0.0:
            return 0.0, 0.0
        from intraday.reference.gaussian_reference import _prob_hit_up
        return _prob_hit_up(spot, 103.0, sqrt(W)), 0.0
    return None


def _levels(cell: C.Cell):
    return {"analytical": (0,), "mc_rqmc": (0, 1)}.get(cell.engine, (0, 1, 2))


def _decided_now(ctx) -> float:
    """Cash the lifecycle fixes and pays exactly at the valuation instant (a continuous barrier found hit by the latest
    spot): the route reports it as paid, the reference as part of the value now."""
    state = ctx.numerical.lifecycle_state
    if state is None:
        return 0.0
    return float(sum(cf.amount for cf in state.ledger.cashflows if cf.payment_time == 0.0))


def run_cell(cell: C.Cell) -> CellResult:
    ctx, _sw = build_context(cell)
    budget = budgets.price_budget(C.notional(cell.product))
    reference = reference_for(cell, ctx)
    if reference is None:
        return CellResult(cell, None, None, None, budget, False, "inconclusive", "no independent reference")
    ref, unc = reference
    from quantark.asset.equity.engine.settlement_support import pending_receivable_pv
    state = ctx.numerical.lifecycle_state
    fixed_value = (float(pending_receivable_pv(state, ctx.pricing_env)) if state is not None else 0.0) + _decided_now(ctx)
    prices, numerics = [], []

    def record():
        errors = [abs(p - ref) for p in prices]
        refinement = {"prices": prices, "errors": errors, "numerical": numerics}
        return errors, dict(cell=cell, route_price=prices[0], ref_price=ref, ref_uncertainty=unc, budget=budget,
                            numerical=numerics[0], refinement=refinement)

    capped = None
    for level in _levels(cell):
        if cell.engine == "pde" and level > 0 and pde_level_cells(numerics[0], level) > LADDER_MAX_GRID_CELLS:
            capped = (level, pde_level_cells(numerics[0], level))
            break
        engine = engine_for(cell, level, numerics[0] if numerics else None)
        try:
            out = route_for(ctx, engine).price(ctx, engine)
        except CapabilityError as exc:
            return CellResult(cell, None, ref, unc, budget, False, "unsupported", str(exc))
        prices.append(out.contingent_pv + fixed_value)
        numerics.append({k: v for k, v in out.numerical.items() if isinstance(v, (int, float, str, bool, type(None)))})
        if cell.engine == "pde" and level == 0 and numerics[0].get("resolution") == "unqualified":
            # the route itself declines an accuracy claim: the diffusion layer is below the grid in space or time
            return CellResult(passed=False, status="unqualified", reason=numerics[0].get("resolution_reason", ""), **record()[1])
    tol = budget + K_REF * unc
    errors, base = record()
    if capped is not None:
        base["refinement"]["capped"] = {"level": capped[0], "grid_cells": capped[1], "cap": LADDER_MAX_GRID_CELLS}
        return CellResult(passed=False, status="unqualified",
                          reason=f"refinement ladder capped by the grid memory budget at level {capped[0]} "
                                 f"(~{capped[1]:.2e} cells > {LADDER_MAX_GRID_CELLS:.0e}); errors so far {errors}", **base)

    if cell.engine == "analytical":
        ok = errors[0] <= tol
        return CellResult(passed=ok, status="passed" if ok else "failed",
                          reason="" if ok else f"exact route off its closed form by {errors[0]:.3e}", **base)
    if cell.engine == "mc_rqmc":
        se = [n.get("std_error") or 0.0 for n in numerics]
        if any(e > budget + 4.0 * s + K_REF * unc for e, s in zip(errors, se)):
            return CellResult(passed=False, status="failed", reason=f"MC bias beyond noise: errors {errors}, se {se}", **base)
        if K_REF * se[0] > budget:
            return CellResult(passed=False, status="unqualified", reason=f"sampling uncertainty {se[0]:.2e} above budget", **base)
        ok = errors[0] <= tol + K_REF * se[0] and abs(prices[-1] - prices[-2]) <= K_REF * se[-1]
        return CellResult(passed=ok, status="passed" if ok else "unqualified", **base)
    ladder_agrees = abs(prices[-1] - prices[-2]) <= tol
    if errors[0] <= tol and ladder_agrees:
        return CellResult(passed=True, status="passed", **base)
    converging = errors[-1] <= tol or (errors[2] < errors[1] < errors[0])
    if not converging:
        # One grid per level samples ONE position of each barrier inside its cell, and doubling moves that phase: a
        # phase-dependent error zig-zags down the ladder. Decide on the envelope over a few phases per level instead.
        envelope = []
        for level in _levels(cell):
            shifted = [prices[level]]
            for shift in PHASE_SHIFTS:
                engine = engine_for(cell, level, numerics[0], phase_shift=shift)
                shifted.append(route_for(ctx, engine).price(ctx, engine).contingent_pv + fixed_value)
            envelope.append(max(abs(p - ref) for p in shifted))
        base["refinement"]["phase_envelope"] = envelope
        if not (envelope[-1] <= tol or envelope[2] < envelope[1] < envelope[0]):
            return CellResult(passed=False, status="failed",
                              reason=f"refinement ladder does not approach the reference: errors {errors}, "
                                     f"phase envelope {envelope}", **base)
        return CellResult(passed=False, status="unqualified",
                          reason=f"discretisation error {errors[0]:.2e} above tolerance {tol:.2e} at default settings; "
                                 f"phase-dependent (barrier position in its cell), phase envelope converges {envelope}",
                          **base)
    return CellResult(passed=False, status="unqualified",
                      reason=f"discretisation error {errors[0]:.2e} above tolerance {tol:.2e} at default settings; ladder converges",
                      **base)


def _jsonable(result: CellResult) -> dict:
    d = asdict(result)
    d["cell"] = {**asdict(result.cell), "horizon": int(result.cell.horizon.total_seconds()), "id": result.cell.id}
    return d


def append_result(result: CellResult, jsonl_path) -> None:
    """One line per cell; xdist workers append to the same file (small single writes in append mode)."""
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(result), sort_keys=True, default=str) + "\n")


def aggregate(jsonl_path, json_path, *, git_sha=None, wall_time_s=None) -> dict:
    """Collect the appended cells into the evidence JSON and return per-status counts.

    ``git_sha`` names the commit the run executed (HEAD may have moved since).
    """
    with open(jsonl_path, encoding="utf-8") as handle:
        rows = {row["cell"]["id"]: row for row in (json.loads(line) for line in handle if line.strip())}
    sha = git_sha
    if sha is None:
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            sha = None
    payload = {
        "schema": "intraday-gate-c/1", "machine": platform.platform(), "git_sha": sha, "wall_time_s": wall_time_s,
        "budgets": {"price_abs_per_notional": budgets.PRICE_ABS_PER_NOTIONAL, "reference_multiplier": K_REF},
        "reference": {"method": "dense_gaussian_backward", "points": list(GATE_C_POINTS)},
        "cells": [rows[k] for k in sorted(rows)],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    counts: dict = {}
    for row in rows.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def write_results(results, path) -> None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        sha = None
    payload = {
        "schema": "intraday-gate-c/1",
        "machine": platform.platform(),
        "git_sha": sha,
        "budgets": {"price_abs_per_notional": budgets.PRICE_ABS_PER_NOTIONAL, "reference_multiplier": K_REF},
        "reference": {"method": "dense_gaussian_backward", "points": list(GATE_C_POINTS)},
        "cells": [_jsonable(r) for r in sorted(results, key=lambda r: r.cell.id)],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)
