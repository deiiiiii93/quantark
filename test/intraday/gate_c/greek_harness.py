"""Gate C for Greeks: point derivatives and desk moves of every route against the independent reference.

One group = (product, profile, horizon, spot offset, barrier); every engine of the product is judged in the group,
so the reference is solved once. Measures per engine:

  point_delta / point_gamma  route.point_greeks against the reference derivatives, on the route's refinement ladder
  desk_delta / desk_gamma    the route's configured finite spot move against the SAME move on the reference
  point_vega / point_rho / point_dividend_rho   (QUAD V2, analytical) the bump-limit ladder h in {4, 2, 1, 1/2} h0
                             must converge, and the production-bump difference must match the reference's difference
  desk_vega / desk_rho / desk_dividend_rho      (QUAD V2, analytical, MC) the daily one-point moves against the SAME
                             moves on the reference
  desk_theta                 (QUAD V2, analytical, MC) the default one-hour frozen-market roll (clamped to the segment)
                             against the same roll on the reference
  point_theta                (QUAD V2, analytical, MC) the runtime's second-order stencil against the reference's
                             time derivative from the backward equation, r V - (r - q) S Delta - 1/2 dW/dtau S^2 Gamma

Every cell records the engine's ``accuracy_settings``: a certificate is bound to the configuration that earned it.

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
from quantark.intraday.capability import accuracy_settings
from quantark.intraday.greeks import (DEFAULT_THETA_STEP, bump_config_for, cell_price, desk_bump_envs, point_proxy_bump,
                                      point_proxy_difference, point_theta_estimate, resolve_theta_step, _spot_env,
                                      with_pricing_env)
from quantark.intraday.roll import roll_context
from quantark.intraday.timestamp import SECONDS_PER_YEAR

from intraday.gate_c import cells as C
from intraday.gate_c.harness import (LADDER_MAX_GRID_CELLS, PHASE_SHIFTS, SIGMA_INNER, _levels, build_context, engine_for,
                                     pde_level_cells, release_pde_memos)
from intraday.reference import budgets
from intraday.reference.budgets import GATE_C_POINTS, REFERENCE_UNCERTAINTY_MULTIPLIER as K_REF
from intraday.reference.gaussian_reference import _solve_snowball, reference_digital, reference_snowball

GREEK_PROFILES = ("desk", "sessions_only")
#: The price ladder's horizons plus the daily regime up to the fixture's fixing gap (its fifth KO fixing is 30 days
#: before the sixth, so 29 days is the longest horizon with the same confirmed history).
GREEK_HORIZONS = (timedelta(days=29), timedelta(days=14), timedelta(days=7), timedelta(days=3)) + C.HORIZONS
#: Engines swept for the desk vega/rho/dividend-rho moves and for theta. PDE is not: its spot moves already miss the
#: budget at most horizons, and a sweep that cannot pass would only buy grid memory. MC is not: at the certified path
#: count the summed standard errors of a desk move's two prices alone exceed the budget (its desk delta and gamma
#: cells record exactly that), and rescaling a one-basis-point move to +1% multiplies them a hundredfold.
MOVE_ENGINES = ("quad_v2", "analytical")
DESK_MOVES = (("vega", "vol_up"), ("rho", "rate_up"), ("dividend_rho", "div_up"))
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
    return [GreekGroup(p, prof, h, o, b) for p in GREEK_ENGINES for prof in GREEK_PROFILES for h in GREEK_HORIZONS
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
    #: A measure's own knob, bound into its certificate (a desk theta's requested roll).
    measure_settings: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GreekCellResult:
    cell: C.Cell
    route_name: str
    measures: Tuple[MeasureResult, ...]
    settings: dict = field(default_factory=dict)          # accuracy_settings of the level-0 engine every measure used
    market: dict = field(default_factory=dict)            # the families and levels the cell ran on (recorded, not keyed)


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


def _move_uncertainty(moved, base) -> float:
    """Richardson uncertainty of a MOVE, from the move's own levels.

    Both prices of a small move are solved on the same grid levels, so most of their discretisation error cancels in
    the difference; summing the two prices' uncertainties would bound that cancellation away. A one-basis-point rate
    move rescaled to +1% amplifies the sum a hundredfold -- past the budget of a move the reference resolves well.
    """
    return _uncertainty([m - b for m, b in zip(moved, base)])


# ----------------------------------------------------------------------------------------------------------------------
# point delta / gamma
def _point_ladder(cell, ctx, settled=None):
    """[(PointGreeks, numerical)] per refinement level; PDE levels double the grid the route solved at level 0 and
    stop at the grid memory budget (a capped ladder has fewer than three levels). ``settled(pg)`` may stop the ladder
    after level 0 when that level alone already decides every status (refining cannot change it)."""
    out, level0 = [], None
    for level in _levels(cell):
        if cell.engine == "pde" and level > 0 and pde_level_cells(level0, level) > LADDER_MAX_GRID_CELLS:
            break
        engine = engine_for(cell, level, level0)
        route = route_for(ctx, engine)
        pg = route.point_greeks(ctx, engine, certify=False)      # the ladder produces certificates; it never reads them
        numerical = {}
        if cell.engine == "pde":
            release_pde_memos()
            numerical = {k: v for k, v in route.price(ctx, engine).numerical.items()
                         if isinstance(v, (int, float, str, bool, type(None)))}
            release_pde_memos()
            level0 = level0 or numerical
        out.append((pg, numerical))
        if pg.status != "ok" or (level == 0 and settled is not None and settled(pg)):
            break
    return out


def _mc_bias_settled(delta, u_delta, gamma, u_gamma, spot, notional):
    """Whether an MC level-0 paired difference is already beyond its noise on BOTH measures.

    ``_classify_point`` then reports both unqualified on that level alone -- a finite bump's truncation that more
    paths cannot remove -- so the 4x-path level would be spent on a status it cannot change.
    """
    def beyond(value, ref, unc, se, budget):
        return isfinite(ref) and value is not None and abs(value - ref) > budget + 4.0 * se + K_REF * unc

    def settled(pg):
        if pg.status != "ok" or not (isfinite(delta) and isfinite(gamma)):
            return False
        return (beyond(pg.delta, delta, u_delta, pg.uncertainty.get("delta", 0.0), budgets.delta_budget(delta, spot, notional))
                and beyond(pg.gamma, gamma, u_gamma, pg.uncertainty.get("gamma", 0.0),
                           budgets.gamma_budget(gamma, spot, notional)))
    return settled


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
            # the paired estimator is a central difference at a FINITE relative bump: a bias beyond its noise is that
            # bump's truncation error against the derivative, an honest miss rather than a defect
            return MeasureResult(status="unqualified", ladder=lad, **record,
                                 reason=f"finite-bump truncation beyond the sampling noise: errors {errors}, se {se}")
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
                spg = route_for(ctx, engine).point_greeks(ctx, engine, certify=False)
                release_pde_memos()
                shifted.append(getattr(spg, name) if spg.status == "ok" else float("inf"))
            envelope.append(max(abs(v - ref_value) for v in shifted))
        lad["phase_envelope"] = envelope
        if envelope[-1] <= tol or envelope[2] < envelope[1] < envelope[0]:
            return MeasureResult(status="unqualified", ladder=lad,
                                 reason=f"phase-dependent discretisation error; phase envelope converges {envelope}", **record)
    return MeasureResult(status="failed", reason=f"refinement ladder does not approach the reference: errors {errors}",
                         ladder=lad, **record)


def _cached_levels(cache, key, c, product: str, points):
    """Reference price levels of context ``c``, solved once per group: every engine differences the same moves."""
    if key not in cache:
        cache[key] = _reference_price_levels(c, product, points)
    return cache[key]


# ----------------------------------------------------------------------------------------------------------------------
# desk moves
def _desk(cell, ctx, engine, ref_points, notional, cache=None) -> Tuple[MeasureResult, MeasureResult]:
    bc = bump_config_for(engine)
    spot = float(ctx.spot)
    h_d = bc.spot_bump
    h_g = bc.gamma_spot_bump if getattr(bc, "gamma_spot_bump", None) else bc.spot_bump
    route = route_for(ctx, engine)
    factors = {"base": 1.0, "d_up": 1.0 + h_d, "d_dn": 1.0 - h_d, "g_up": 1.0 + h_g, "g_dn": 1.0 - h_g}
    route_px, route_se, ref_levels = {}, {}, {}
    for key, factor in factors.items():
        c = ctx if key == "base" else with_pricing_env(ctx, _spot_env(ctx.pricing_env, factor), f"desk_{key}")
        out = route.price(c, engine)
        route_px[key] = out.contingent_pv + _pending(c)
        route_se[key] = float(out.numerical.get("std_error") or 0.0)
        out = None
        release_pde_memos()
        ref_levels[key] = _cached_levels({} if cache is None else cache, ("spot", factor), c, cell.product, ref_points)
    ref_px = {key: levels[-1] for key, levels in ref_levels.items()}
    results = []
    for name, (up, dn), h in (("delta", ("d_up", "d_dn"), h_d), ("gamma", ("g_up", "g_dn"), h_g)):
        if name == "delta":
            scale = 2.0 * spot * h

            def combo(px):
                return (px[up] - px[dn]) / scale
            se = (route_se[up] + route_se[dn]) / scale
            budget = budgets.delta_budget(combo(ref_px), spot, notional)
        else:
            scale = (spot * h) ** 2

            def combo(px):
                return (px[up] - 2.0 * px["base"] + px[dn]) / scale
            se = (route_se[up] + 2.0 * route_se["base"] + route_se[dn]) / scale
            budget = budgets.gamma_budget(combo(ref_px), spot, notional)
        # the move's OWN Richardson levels: its prices share each reference grid, so summing their separate
        # uncertainties would bound the cancellation away (see _move_uncertainty)
        spread = _uncertainty([combo(dict(zip(ref_levels, level))) for level in zip(*ref_levels.values())])
        route_v, ref_v = combo(route_px), combo(ref_px)
        record = dict(measure=f"desk_{name}", route=route_v, ref=ref_v, ref_uncertainty=spread, budget=budget,
                      ladder={"bump": h, "route_std_error": se})
        err = abs(route_v - ref_v)
        if K_REF * spread > budget:
            results.append(MeasureResult(status="inconclusive", reason=f"reference uncertainty {spread:.2e} exceeds the budget", **record))
        elif K_REF * se > budget:
            # judged BEFORE the pass: a standard error above the budget would otherwise widen the tolerance into one
            results.append(MeasureResult(status="unqualified", reason=f"sampling uncertainty {se:.2e} above budget", **record))
        elif err <= budget + K_REF * (spread + se):
            results.append(MeasureResult(status="passed", **record))
        elif cell.engine == "pde":
            results.append(MeasureResult(status="unqualified", reason=f"grid error {err:.2e} on the finite move (desk moves have no ladder)", **record))
        else:
            results.append(MeasureResult(status="failed", reason=f"finite move off the reference by {err:.3e}", **record))
    return tuple(results)


def _classify_move(cell, record, err, budget, spread, se) -> MeasureResult:
    """A finite move against the same move on the reference (the desk-move statuses of ``_desk``)."""
    if K_REF * spread > budget:
        return MeasureResult(status="inconclusive", reason=f"reference uncertainty {spread:.2e} exceeds the budget", **record)
    if K_REF * se > budget:
        return MeasureResult(status="unqualified", reason=f"sampling uncertainty {se:.2e} above budget", **record)
    if err <= budget + K_REF * (spread + se):
        return MeasureResult(status="passed", **record)
    if cell.engine == "quad_v2":
        return MeasureResult(status="unqualified", reason=f"quadrature error {err:.2e} on the finite move", **record)
    return MeasureResult(status="failed", reason=f"finite move off the reference by {err:.3e}", **record)


def _route_value(route, engine, c):
    """(price incl. pending receivables, standard error) of one context on the route."""
    out = route.price(c, engine)
    return out.contingent_pv + _pending(c), float(out.numerical.get("std_error") or 0.0)


def _desk_moves(cell, ctx, engine, ref_points, notional, cache) -> Tuple[MeasureResult, ...]:
    """Desk vega / rho / dividend rho: the daily one-point moves on the route and on the reference."""
    bc = bump_config_for(engine)
    route = route_for(ctx, engine)
    envs = desk_bump_envs(ctx, engine, tuple(name for name, _ in DESK_MOVES))
    base_px, base_se = _route_value(route, engine, ctx)
    base_levels = _cached_levels(cache, ("spot", 1.0), ctx, cell.product, ref_points)
    results = []
    for name, bump_id in DESK_MOVES:
        c = with_pricing_env(ctx, envs[bump_id], f"desk_{bump_id}")
        px, se = _route_value(route, engine, c)
        bump = {"vega": bc.vol_bump, "rho": bc.rate_bump, "dividend_rho": bc.div_bump}[name]
        levels = _cached_levels(cache, (bump_id, bump), c, cell.product, ref_points)
        # the published unit: raw PnL per +vol_bump for vega, rescaled to +1% for rho and dividend rho
        scale = 1.0 if name == "vega" else 0.01 / (bc.rate_bump if name == "rho" else bc.div_bump)
        route_v, ref_v = (px - base_px) * scale, (levels[-1] - base_levels[-1]) * scale
        spread = _move_uncertainty(levels, base_levels) * scale
        budget = budgets.desk_move_budget(ref_v, notional)
        record = dict(measure=f"desk_{name}", route=route_v, ref=ref_v, ref_uncertainty=spread, budget=budget,
                      ladder={"bump": bump, "route_std_error": (se + base_se) * scale})
        results.append(_classify_move(cell, record, abs(route_v - ref_v), budget, spread, (se + base_se) * scale))
    return tuple(results)


def _desk_theta(cell, ctx, engine, ref_points, notional, cache) -> MeasureResult:
    """The default desk roll: one hour on the frozen market, clamped to the segment, on the route and the reference."""
    route = route_for(ctx, engine)
    step = resolve_theta_step(ctx, None, "hour")
    knobs = {"theta_step_s": DEFAULT_THETA_STEP.total_seconds()}
    if step.actual is None:
        return MeasureResult("desk_theta", "undefined", reason=step.reason, measure_settings=knobs)
    rolled = roll_context(ctx, (ctx.valuation_timestamp + step.actual))
    received = float(rolled.numerical.paid_cash) - float(ctx.numerical.paid_cash)
    p0, se0 = _route_value(route, engine, ctx)
    p1, se1 = _route_value(route, engine, rolled)
    l0 = _cached_levels(cache, ("spot", 1.0), ctx, cell.product, ref_points)
    l1 = _cached_levels(cache, ("roll", step.actual.total_seconds()), rolled, cell.product, ref_points)
    route_v = (p1 + received - p0) / step.divisor
    ref_v = (l1[-1] + received - l0[-1]) / step.divisor
    spread = _move_uncertainty(l1, l0) / step.divisor
    se = (se0 + se1) / step.divisor
    budget = budgets.theta_budget(ref_v, notional)
    record = dict(measure="desk_theta", route=route_v, ref=ref_v, ref_uncertainty=spread, budget=budget,
                  ladder={"step_s": step.actual.total_seconds(), "route_std_error": se}, measure_settings=knobs)
    return _classify_move(cell, record, abs(route_v - ref_v), budget, spread, se)


def _point_theta(cell, ctx, engine, reference, notional) -> MeasureResult:
    """The runtime stencil against the backward equation evaluated on the reference's value, delta and gamma.

    Between events the frozen-market value solves dV/dt = r V - (r - q) S dV/dS - 1/2 (dW/dtau) S^2 d2V/dS2 with the
    instantaneous forward rate, carry and variance rate at the valuation instant; a fixed pending receivable grows at
    r. The Gate C market is flat, so r and q are its levels and dW/dtau = sigma^2 x the clock's slope at 0+.
    """
    route = route_for(ctx, engine)
    price_now, se_now = _route_value(route, engine, ctx)
    estimate = point_theta_estimate(ctx, engine, price_base=price_now, unit="hour")
    record = dict(measure="point_theta")
    if estimate is None:
        return MeasureResult(status="unqualified", reason="no admissible stencil step in this segment", **record)
    value, h, _exact = estimate
    ref_price, ref_delta, ref_gamma, u_price, u_delta, u_gamma = reference
    if not (isfinite(ref_delta) and isfinite(ref_gamma)):
        return MeasureResult(status="inconclusive", route=value, reason="the reference has no spot derivatives here",
                             **record)
    r, q = C.market(cell.product)
    spot = float(ctx.spot)
    rate_w = SIGMA_INNER ** 2 * float(ctx.time_map.initial_slope())
    per_hour = 3600.0 / SECONDS_PER_YEAR
    total = ref_price + _pending(ctx)
    ref_v = (r * total - (r - q) * spot * ref_delta - 0.5 * rate_w * spot * spot * ref_gamma) * per_hour
    unc = (abs(r) * u_price + abs(r - q) * spot * u_delta + 0.5 * rate_w * spot * spot * u_gamma) * per_hour
    budget = budgets.theta_budget(ref_v, notional)
    # MC: the stencil's weights (3, 4, 1) on three prices of about the base's standard error, over its 2h, per hour
    se = 8.0 * se_now / (2.0 * h.total_seconds()) * 3600.0
    record.update(route=value, ref=ref_v, ref_uncertainty=unc, budget=budget,
                  ladder={"step_s": h.total_seconds(), "route_std_error": se})
    err = abs(value - ref_v)
    if K_REF * unc > budget:
        return MeasureResult(status="inconclusive", reason=f"reference uncertainty {unc:.2e} exceeds the budget", **record)
    if K_REF * se > budget:
        return MeasureResult(status="unqualified", reason=f"sampling uncertainty {se:.2e} above budget", **record)
    if err <= budget + K_REF * (unc + se):
        return MeasureResult(status="passed", **record)
    if cell.engine == "quad_v2":
        return MeasureResult(status="unqualified", reason=f"stencil on the quadrature off by {err:.2e}", **record)
    return MeasureResult(status="failed", reason=f"stencil off the backward-equation theta by {err:.3e}", **record)


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
def run_greek_group(group: GreekGroup, *, reference_points=GATE_C_POINTS, proxies: bool = True, moves: bool = True):
    """Every engine of the group's product: a GreekCellResult per engine (reference solved once)."""
    base_cell = group.cell(GREEK_ENGINES[group.product][0])
    ctx, _sw = build_context(base_cell)
    notional = C.notional(group.product)
    spot = float(ctx.spot)
    if ctx.numerical.terminated:
        reference = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        reference = _reference(ctx, group.product, reference_points)
    _price, delta, gamma, _up, ud, ug = reference
    cache: dict = {}
    market = {"vol_surface": type(ctx.request.pricing_env.vol_surface).__name__,
              "rate_curve": type(ctx.request.pricing_env.rate_curve).__name__,
              "div_yield": type(ctx.request.pricing_env.div_yield).__name__,
              "levels": {"vol": SIGMA_INNER, "r": C.market(group.product)[0], "q": C.market(group.product)[1]}}
    results = []
    for engine_key in GREEK_ENGINES[group.product]:
        cell = group.cell(engine_key)
        route_name = ROUTE_NAMES[(group.product, engine_key)]
        settings = accuracy_settings(engine_for(cell, 0))
        settled = _mc_bias_settled(delta, ud, gamma, ug, spot, notional) if engine_key == "mc_rqmc" else None
        try:
            ladder = _point_ladder(cell, ctx, settled)
        except CapabilityError as exc:
            results.append(GreekCellResult(cell, route_name, (MeasureResult("point_delta", "unsupported", reason=str(exc)),
                                                              MeasureResult("point_gamma", "unsupported", reason=str(exc))),
                                           settings, market))
            continue
        measures = [
            _classify_point(cell, ctx, "delta", ladder, delta, ud,
                            budgets.delta_budget(delta, spot, notional) if isfinite(delta) else float("nan")),
            _classify_point(cell, ctx, "gamma", ladder, gamma, ug,
                            budgets.gamma_budget(gamma, spot, notional) if isfinite(gamma) else float("nan")),
        ]
        measures.extend(_desk(cell, ctx, engine_for(cell, 0), reference_points, notional, cache))
        if proxies and engine_key in PROXY_ENGINES:
            measures.extend(_proxy(cell, ctx, engine_for(cell, 0), name, notional) for name in PROXIES)
        if moves and engine_key in MOVE_ENGINES:
            measures.extend(_desk_moves(cell, ctx, engine_for(cell, 0), reference_points, notional, cache))
            measures.append(_desk_theta(cell, ctx, engine_for(cell, 0), reference_points, notional, cache))
            measures.append(_point_theta(cell, ctx, engine_for(cell, 0), reference, notional))
        results.append(GreekCellResult(cell, route_name, tuple(measures), settings, market))
    return results


def _jsonable(result: GreekCellResult) -> dict:
    return {"cell": {**asdict(result.cell), "horizon": int(result.cell.horizon.total_seconds()), "id": result.cell.id},
            "route": result.route_name, "product": PRODUCT_NAMES[result.cell.product],
            "settings": result.settings, "market": result.market,
            "measures": [asdict(m) for m in result.measures]}


def append_greek_results(results, jsonl_path) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps(_jsonable(r), sort_keys=True, default=str) + "\n")


def demonstrated(rows) -> list:
    """One row per maximal window of consecutive swept horizons at which EVERY cell of a family passed.

    The family is (product, route, measure, monitoring, profile, engine settings, measure settings): a certificate
    covers only the configurations its cells ran. ``horizon_s``/``horizon_max_s`` bound a window of consecutive
    swept horizons with every cell passing (``undefined`` agreeing with the reference counts); beyond either end, and
    across a horizon where any cell missed, the evidence is silent (review 2026-09-16 findings 4 and R4). The spot
    offsets, barriers and market families are recorded so a reader can see the domain the claim rests on.
    """
    import json as _json
    from intraday.gate_c.cells import MONITORING, profile as profile_of

    table: dict = {}
    for row in rows:
        cell = row["cell"]
        settings = row.get("settings") or {}
        for m in row["measures"]:
            knobs = m.get("measure_settings") or {}
            key = (row["product"], row["route"], m["measure"], MONITORING[cell["product"]], cell["profile"],
                   _json.dumps(settings, sort_keys=True), _json.dumps(knobs, sort_keys=True))
            ok = m["status"] in ("passed", "undefined")
            entry = table.setdefault(key, {"by_h": {}, "offsets": set(), "barriers": set(), "settings": settings,
                                           "measure_settings": knobs, "markets": []})
            entry["by_h"].setdefault(cell["horizon"], []).append(ok)
            entry["offsets"].add(cell["offset"])
            entry["barriers"].add(cell["barrier"])
            if row.get("market") and row["market"] not in entry["markets"]:
                entry["markets"].append(row["market"])
    out = []
    for (product, route, measure, monitoring, profile_name, _s, _k), entry in sorted(table.items()):
        horizons = sorted(entry["by_h"])
        windows, start = [], None
        for i, h in enumerate(horizons):
            if all(entry["by_h"][h]):
                start = h if start is None else start
                if i == len(horizons) - 1 or not all(entry["by_h"][horizons[i + 1]]):
                    windows.append((start, h))
                    start = None
        for lo, hi in windows:
            out.append({"product": product, "route": route, "measure": measure, "monitoring": monitoring,
                        "profile": profile_name, "profile_identity": list(profile_of(profile_name).identity()),
                        "settings": entry["settings"], "measure_settings": entry["measure_settings"],
                        "horizon_s": lo, "horizon_max_s": hi, "swept_horizons_s": [h for h in horizons if lo <= h <= hi],
                        "offsets": sorted(entry["offsets"]), "barriers": sorted(entry["barriers"]),
                        "markets": entry["markets"]})
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
        "schema": "intraday-gate-c-greeks/2", "machine": platform.platform(), "git_sha": git_sha, "wall_time_s": wall_time_s,
        "budgets": {"delta": [budgets.DELTA_ABS, budgets.DELTA_REL], "gamma": [budgets.GAMMA_ABS, budgets.GAMMA_REL],
                    "move_per_point": [budgets.MOVE_ABS, budgets.MOVE_REL],
                    "theta_per_hour": [budgets.THETA_ABS, budgets.THETA_REL], "reference_multiplier": K_REF},
        "bump_ladder": list(BUMP_LADDER),
        "horizons_s": [int(h.total_seconds()) for h in sorted(GREEK_HORIZONS)],
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
