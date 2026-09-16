"""Gate C for Greeks: point derivatives and desk moves of every route against the independent reference.

One group = (product, profile, horizon, spot offset, barrier); every engine of the product is judged in the group,
so the reference is solved once. Measures per engine:

  point_delta / point_gamma  route.point_greeks against the reference derivatives, on the route's refinement ladder
  desk_delta / desk_gamma    the route's configured finite spot move against the SAME move on the reference
  point_vega / point_rho / point_dividend_rho   (QUAD V2, analytical) the bump-limit ladder h in {4, 2, 1, 1/2} h0
                             must converge, and the production-bump difference must match the reference's difference

Statuses (never better than the evidence): passed, unqualified (the route declines, a converging ladder above
budget, sampling error above budget), undefined (the derivative does not exist and route and reference agree),
inconclusive (the reference's own uncertainty exceeds the budget), unsupported (no route), failed (anything else).
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from math import isfinite
from typing import Optional, Tuple

from quantark.asset.equity.engine.settlement_support import pending_receivable_pv
from quantark.execution.errors import CapabilityError
from quantark.intraday.engines import route_for
from quantark.intraday.greeks import (bump_config_for, cell_price, point_proxy_bump, point_proxy_difference, _spot_env,
                                      with_pricing_env)

from intraday.gate_c import cells as C
from intraday.gate_c.harness import (LADDER_MAX_GRID_CELLS, PHASE_SHIFTS, _levels, build_context, engine_for, pde_level_cells,
                                     release_pde_memos)
from intraday.reference import budgets
from intraday.reference.budgets import GATE_C_POINTS, REFERENCE_UNCERTAINTY_MULTIPLIER as K_REF
from intraday.reference.gaussian_reference import _solve_snowball, reference_digital, reference_snowball

GREEK_PROFILES = ("desk", "sessions_only")
GREEK_ENGINES = {"snowball_discrete_ki": ("quad_v2", "pde", "mc_rqmc"), "digital": ("analytical", "mc_rqmc")}
PROXY_ENGINES = ("quad_v2", "analytical")
PROXIES = ("vega", "rho", "dividend_rho")
BUMP_LADDER = (4.0, 2.0, 1.0, 0.5)
PROXY_REFERENCE_POINTS = (4001, 8001)
ROUTE_NAMES = {("snowball_discrete_ki", "quad_v2"): "QuadV2Route", ("snowball_discrete_ki", "pde"): "PDERoute",
               ("snowball_discrete_ki", "mc_rqmc"): "MCRoute", ("digital", "analytical"): "AnalyticalDigitalRoute",
               ("digital", "mc_rqmc"): "MCRoute"}
PRODUCT_NAMES = {"snowball_discrete_ki": "SnowballOption", "digital": "CashOrNothingDigitalOption"}


@dataclass(frozen=True)
class GreekGroup:
    product: str
    profile: str
    horizon: timedelta
    offset: str
    barrier: str

    @property
    def id(self) -> str:
        return f"greek-{self.product}-{self.profile}-{int(self.horizon.total_seconds())}s-{self.offset}-{self.barrier}"

    def cell(self, engine: str) -> C.Cell:
        return C.Cell(self.product, engine, self.profile, self.horizon, self.offset, self.barrier)


def greek_groups():
    return [GreekGroup(p, prof, h, o, b) for p in GREEK_ENGINES for prof in GREEK_PROFILES for h in C.HORIZONS
            for o in C.SPOT_OFFSETS for b in C.BARRIERS[p]]


def fast_greek_groups():
    return [GreekGroup(p, C.FAST_PROFILE, C.FAST_HORIZON, C.FAST_OFFSET, C.BARRIERS[p][0]) for p in GREEK_ENGINES]


@dataclass(frozen=True)
class MeasureResult:
    measure: str
    status: str
    route: Optional[float] = None
    ref: Optional[float] = None
    ref_uncertainty: Optional[float] = None
    budget: Optional[float] = None
    reason: str = ""
    ladder: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GreekCellResult:
    cell: C.Cell
    route_name: str
    measures: Tuple[MeasureResult, ...]


# ----------------------------------------------------------------------------------------------------------------------
# references
def _reference(ctx, product: str, points):
    """(price, delta, gamma, unc_price, unc_delta, unc_gamma) of the remaining claim, excluding pending receivables."""
    if product == "digital":
        r = reference_digital(ctx)
    else:
        r = reference_snowball(ctx, points=points)
    return r.price, r.delta, r.gamma, r.uncertainty_price, r.uncertainty_delta, r.uncertainty_gamma


def _pending(ctx) -> float:
    state = ctx.numerical.lifecycle_state
    return float(pending_receivable_pv(state, ctx.pricing_env)) if state is not None else 0.0


def _reference_price_levels(ctx, product: str, points):
    """Reference price (+ pending receivables on the context's own environment) on each grid level."""
    if product == "digital":
        return [reference_digital(ctx).price + _pending(ctx)]
    if ctx.numerical.terminated:
        return [_pending(ctx) for _ in points]
    return [_solve_snowball(ctx, p, 8.0)[0] + _pending(ctx) for p in points]


def _uncertainty(levels) -> float:
    return abs(levels[-1] - levels[-2]) / 3.0 if len(levels) > 1 else 0.0


# ----------------------------------------------------------------------------------------------------------------------
# point delta / gamma
def _point_ladder(cell, ctx):
    """[(PointGreeks, numerical)] per refinement level; PDE levels double the grid the route solved at level 0 and
    stop at the grid memory budget (a capped ladder has fewer than three levels)."""
    out, level0 = [], None
    for level in _levels(cell):
        if cell.engine == "pde" and level > 0 and pde_level_cells(level0, level) > LADDER_MAX_GRID_CELLS:
            break
        engine = engine_for(cell, level, level0)
        route = route_for(ctx, engine)
        pg = route.point_greeks(ctx, engine)
        numerical = {}
        if cell.engine == "pde":
            release_pde_memos()
            numerical = {k: v for k, v in route.price(ctx, engine).numerical.items()
                         if isinstance(v, (int, float, str, bool, type(None)))}
            release_pde_memos()
            level0 = level0 or numerical
        out.append((pg, numerical))
        if pg.status != "ok":
            break
    return out


def _classify_point(cell, ctx, name, ladder, ref_value, ref_unc, budget) -> MeasureResult:
    pg0 = ladder[0][0]
    values = [getattr(pg, name) for pg, _ in ladder]
    record = dict(measure=f"point_{name}", route=values[0], ref=ref_value if isfinite(ref_value) else None,
                  ref_uncertainty=ref_unc if isfinite(ref_unc) else None, budget=budget)
    if pg0.status == "undefined":
        if isfinite(ref_value):
            return MeasureResult(status="failed", reason=f"route undefined where the reference is defined: {pg0.reason}",
                                 **record)
        return MeasureResult(status="undefined", reason=pg0.reason, **record)
    if not isfinite(ref_value):
        return MeasureResult(status="failed" if pg0.status == "ok" else pg0.status,
                             reason="the reference derivative does not exist here" + (f"; route: {pg0.reason}" if pg0.reason else ""),
                             **record)
    if pg0.status != "ok":
        return MeasureResult(status=pg0.status, reason=pg0.reason, **record)
    if K_REF * ref_unc > budget:
        return MeasureResult(status="inconclusive", reason=f"reference uncertainty {ref_unc:.2e} exceeds the budget", **record)
    tol = budget + K_REF * ref_unc
    if any(pg.status != "ok" for pg, _ in ladder):
        return MeasureResult(status="unqualified", reason=f"a refined level declines: {ladder[-1][0].reason}",
                             ladder={"values": values}, **record)
    errors = [abs(v - ref_value) for v in values]
    if cell.engine == "mc_rqmc":
        se = [pg.uncertainty.get(name, 0.0) for pg, _ in ladder]
        lad = {"values": values, "errors": errors, "std_errors": se}
        if any(e > budget + 4.0 * s + K_REF * ref_unc for e, s in zip(errors, se)):
            return MeasureResult(status="failed", reason=f"MC bias beyond noise: errors {errors}, se {se}", ladder=lad, **record)
        if K_REF * se[0] > budget:
            return MeasureResult(status="unqualified", reason=f"sampling uncertainty {se[0]:.2e} above budget", ladder=lad, **record)
        ok = errors[0] <= tol + K_REF * se[0] and abs(values[-1] - values[-2]) <= K_REF * se[-1]
        return MeasureResult(status="passed" if ok else "unqualified", ladder=lad, **record)
    lad = {"values": values, "errors": errors}
    if cell.engine == "pde" and len(values) < len(_levels(cell)):
        return MeasureResult(status="unqualified", ladder=lad,
                             reason=f"refinement ladder capped by the grid memory budget after {len(values)} level(s)", **record)
    if len(values) == 1:                      # analytical closed form
        ok = errors[0] <= tol
        return MeasureResult(status="passed" if ok else "failed", reason="" if ok else f"closed form off by {errors[0]:.3e}",
                             ladder=lad, **record)
    if errors[0] <= tol and abs(values[-1] - values[-2]) <= tol:
        return MeasureResult(status="passed", ladder=lad, **record)
    if errors[-1] <= tol or errors[2] < errors[1] < errors[0]:
        return MeasureResult(status="unqualified", reason=f"discretisation error {errors[0]:.2e} above {tol:.2e}; ladder converges",
                             ladder=lad, **record)
    if cell.engine == "pde":
        envelope = []
        for level, (pg, numerical) in enumerate(ladder):
            shifted = [getattr(pg, name)]
            for shift in PHASE_SHIFTS:
                engine = engine_for(cell, level, ladder[0][1], phase_shift=shift)
                spg = route_for(ctx, engine).point_greeks(ctx, engine)
                release_pde_memos()
                shifted.append(getattr(spg, name) if spg.status == "ok" else float("inf"))
            envelope.append(max(abs(v - ref_value) for v in shifted))
        lad["phase_envelope"] = envelope
        if envelope[-1] <= tol or envelope[2] < envelope[1] < envelope[0]:
            return MeasureResult(status="unqualified", ladder=lad,
                                 reason=f"phase-dependent discretisation error; phase envelope converges {envelope}", **record)
    return MeasureResult(status="failed", reason=f"refinement ladder does not approach the reference: errors {errors}",
                         ladder=lad, **record)


# ----------------------------------------------------------------------------------------------------------------------
# desk moves
def _desk(cell, ctx, engine, ref_points, notional) -> Tuple[MeasureResult, MeasureResult]:
    bc = bump_config_for(engine)
    spot = float(ctx.spot)
    h_d = bc.spot_bump
    h_g = bc.gamma_spot_bump if getattr(bc, "gamma_spot_bump", None) else bc.spot_bump
    route = route_for(ctx, engine)
    factors = {"base": 1.0, "d_up": 1.0 + h_d, "d_dn": 1.0 - h_d, "g_up": 1.0 + h_g, "g_dn": 1.0 - h_g}
    route_px, route_se, ref_px, ref_unc = {}, {}, {}, {}
    for key, factor in factors.items():
        c = ctx if key == "base" else with_pricing_env(ctx, _spot_env(ctx.pricing_env, factor), f"desk_{key}")
        out = route.price(c, engine)
        route_px[key] = out.contingent_pv + _pending(c)
        route_se[key] = float(out.numerical.get("std_error") or 0.0)
        out = None
        release_pde_memos()
        levels = _reference_price_levels(c, cell.product, ref_points)
        ref_px[key], ref_unc[key] = levels[-1], _uncertainty(levels)
    results = []
    for name, (up, dn), h in (("delta", ("d_up", "d_dn"), h_d), ("gamma", ("g_up", "g_dn"), h_g)):
        if name == "delta":
            scale = 2.0 * spot * h

            def combo(px):
                return (px[up] - px[dn]) / scale
            spread = (ref_unc[up] + ref_unc[dn]) / scale
            se = (route_se[up] + route_se[dn]) / scale
            budget = budgets.delta_budget(combo(ref_px), spot, notional)
        else:
            scale = (spot * h) ** 2

            def combo(px):
                return (px[up] - 2.0 * px["base"] + px[dn]) / scale
            spread = (ref_unc[up] + 2.0 * ref_unc["base"] + ref_unc[dn]) / scale
            se = (route_se[up] + 2.0 * route_se["base"] + route_se[dn]) / scale
            budget = budgets.gamma_budget(combo(ref_px), spot, notional)
        route_v, ref_v = combo(route_px), combo(ref_px)
        record = dict(measure=f"desk_{name}", route=route_v, ref=ref_v, ref_uncertainty=spread, budget=budget,
                      ladder={"bump": h, "route_std_error": se})
        err = abs(route_v - ref_v)
        if K_REF * spread > budget:
            results.append(MeasureResult(status="inconclusive", reason=f"reference uncertainty {spread:.2e} exceeds the budget", **record))
        elif err <= budget + K_REF * (spread + se):
            results.append(MeasureResult(status="passed", **record))
        elif cell.engine == "mc_rqmc" and K_REF * se > budget:
            results.append(MeasureResult(status="unqualified", reason=f"sampling uncertainty {se:.2e} above budget", **record))
        elif cell.engine == "pde":
            results.append(MeasureResult(status="unqualified", reason=f"grid error {err:.2e} on the finite move (desk moves have no ladder)", **record))
        else:
            results.append(MeasureResult(status="failed", reason=f"finite move off the reference by {err:.3e}", **record))
    return tuple(results)


# ----------------------------------------------------------------------------------------------------------------------
# bump-limit ladders of the point proxies
def _proxy(cell, ctx, engine, name, notional) -> MeasureResult:
    h0 = point_proxy_bump(ctx, name)
    diffs = [point_proxy_difference(ctx, name, k * h0, lambda c: cell_price(c, engine)) for k in BUMP_LADDER]
    ref_levels = []
    for level in range(len(PROXY_REFERENCE_POINTS) if cell.product != "digital" else 1):
        points = (PROXY_REFERENCE_POINTS[level],)
        ref_levels.append(point_proxy_difference(ctx, name, h0, lambda c: _reference_price_levels(c, cell.product, points)[0]))
    ref_v, ref_unc = ref_levels[-1], _uncertainty(ref_levels)
    route_v = diffs[BUMP_LADDER.index(1.0)]
    budget = budgets.move_budget(ref_v, notional)
    tol = budget + K_REF * ref_unc
    d4, d2, d1, dh = diffs
    ratio = (d4 - d2) / (d2 - d1) if d2 != d1 else float("inf")
    converged = abs(dh - d1) <= tol or 3.0 <= ratio <= 5.0
    record = dict(measure=f"point_{name}", route=route_v, ref=ref_v, ref_uncertainty=ref_unc, budget=budget,
                  ladder={"bumps": [k * h0 for k in BUMP_LADDER], "differences": diffs, "richardson_ratio": ratio,
                          "reference_levels": ref_levels})
    if K_REF * ref_unc > budget:
        return MeasureResult(status="inconclusive", reason=f"reference uncertainty {ref_unc:.2e} exceeds the budget", **record)
    if not converged:
        return MeasureResult(status="unqualified", reason=f"bump ladder does not converge (ratio {ratio:.3g})", **record)
    if abs(route_v - ref_v) <= tol:
        return MeasureResult(status="passed", **record)
    return MeasureResult(status="failed", reason=f"converged difference off the reference by {abs(route_v - ref_v):.3e}", **record)


# ----------------------------------------------------------------------------------------------------------------------
def run_greek_group(group: GreekGroup, *, reference_points=GATE_C_POINTS, proxies: bool = True):
    """Every engine of the group's product: a GreekCellResult per engine (reference solved once)."""
    base_cell = group.cell(GREEK_ENGINES[group.product][0])
    ctx, _sw = build_context(base_cell)
    notional = C.notional(group.product)
    spot = float(ctx.spot)
    if ctx.numerical.terminated:
        _price, delta, gamma, _up, ud, ug = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        _price, delta, gamma, _up, ud, ug = _reference(ctx, group.product, reference_points)
    results = []
    for engine_key in GREEK_ENGINES[group.product]:
        cell = group.cell(engine_key)
        route_name = ROUTE_NAMES[(group.product, engine_key)]
        try:
            ladder = _point_ladder(cell, ctx)
        except CapabilityError as exc:
            results.append(GreekCellResult(cell, route_name, (MeasureResult("point_delta", "unsupported", reason=str(exc)),
                                                              MeasureResult("point_gamma", "unsupported", reason=str(exc)))))
            continue
        measures = [
            _classify_point(cell, ctx, "delta", ladder, delta, ud,
                            budgets.delta_budget(delta, spot, notional) if isfinite(delta) else float("nan")),
            _classify_point(cell, ctx, "gamma", ladder, gamma, ug,
                            budgets.gamma_budget(gamma, spot, notional) if isfinite(gamma) else float("nan")),
        ]
        measures.extend(_desk(cell, ctx, engine_for(cell, 0), reference_points, notional))
        if proxies and engine_key in PROXY_ENGINES:
            measures.extend(_proxy(cell, ctx, engine_for(cell, 0), name, notional) for name in PROXIES)
        results.append(GreekCellResult(cell, route_name, tuple(measures)))
    return results


def _jsonable(result: GreekCellResult) -> dict:
    return {"cell": {**asdict(result.cell), "horizon": int(result.cell.horizon.total_seconds()), "id": result.cell.id},
            "route": result.route_name, "product": PRODUCT_NAMES[result.cell.product],
            "measures": [asdict(m) for m in result.measures]}


def append_greek_results(results, jsonl_path) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps(_jsonable(r), sort_keys=True, default=str) + "\n")


def demonstrated(rows) -> list:
    """One row per (product, route, measure, monitoring, profile) family the ladder actually swept.

    A certificate covers only the configurations its cells ran, so the key carries the
    whole family and the row carries the horizon WINDOW: ``horizon_s`` is the shortest
    horizon at and above which every cell passed, ``horizon_max_s`` the longest the
    ladder reached. Beyond either end the evidence is silent (review 2026-09-16
    finding 4), and the spot offsets and barriers are recorded so a reader can see the
    domain the claim rests on.
    """
    from intraday.gate_c.cells import MONITORING, profile as profile_of

    table: dict = {}
    for row in rows:
        cell = row["cell"]
        for m in row["measures"]:
            key = (row["product"], row["route"], m["measure"], MONITORING[cell["product"]], cell["profile"])
            ok = m["status"] in ("passed", "undefined")
            entry = table.setdefault(key, {"by_h": {}, "offsets": set(), "barriers": set()})
            entry["by_h"].setdefault(cell["horizon"], []).append(ok)
            entry["offsets"].add(cell["offset"])
            entry["barriers"].add(cell["barrier"])
    out = []
    for (product, route, measure, monitoring, profile_name), entry in sorted(table.items()):
        by_h = entry["by_h"]
        horizon = None
        for h in sorted(by_h, reverse=True):
            if not all(by_h[h]):
                break
            horizon = h
        if horizon is None:
            continue
        out.append({"product": product, "route": route, "measure": measure, "monitoring": monitoring,
                    "profile": profile_name, "profile_identity": list(profile_of(profile_name).identity()),
                    "horizon_s": horizon, "horizon_max_s": max(by_h),
                    "offsets": sorted(entry["offsets"]), "barriers": sorted(entry["barriers"])})
    return out


def aggregate_greeks(jsonl_path, json_path, *, git_sha=None, wall_time_s=None) -> dict:
    with open(jsonl_path, encoding="utf-8") as handle:
        rows = {}
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["cell"]["id"]] = row
    if git_sha is None:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    payload = {
        "schema": "intraday-gate-c-greeks/1", "machine": platform.platform(), "git_sha": git_sha, "wall_time_s": wall_time_s,
        "budgets": {"delta": [budgets.DELTA_ABS, budgets.DELTA_REL], "gamma": [budgets.GAMMA_ABS, budgets.GAMMA_REL],
                    "move_per_point": [budgets.MOVE_ABS, budgets.MOVE_REL], "reference_multiplier": K_REF},
        "bump_ladder": list(BUMP_LADDER),
        "demonstrated": demonstrated(rows.values()),
        "cells": [rows[k] for k in sorted(rows)],
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))  # packaged: compact
    counts: dict = {}
    for row in rows.values():
        for m in row["measures"]:
            counts[m["status"]] = counts.get(m["status"], 0) + 1
    return counts
