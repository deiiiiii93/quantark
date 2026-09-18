"""Intraday Greeks: re-evaluations of the SAME resolved price function.

A bump cell replaces only the numerical environment of the resolved context;
the twin, the time-based ledger and every confirmed or assumed fixing are
shared by identity, so a bumped spot never re-decides an observation. Desk
bumps reuse the daily helpers and conventions exactly (``bump_envs`` and
``GreeksCalculator``): relative central spot bumps, a one-sided raw vega per
``vol_bump`` of the trading-quoted surface, one-sided rho and dividend rho
rescaled to +1%.

Point Greeks are derivatives of that price function at the query spot, taken
by each route's own evidence (kernel derivative, closed form, grid stencil,
paired RQMC). Where the function jumps at the query spot they are undefined;
where the route cannot vouch for its derivative they are unqualified. Point
vega/rho/dividend rho are finite-difference proxies that stay unqualified
(no value) until a Gate C bump-limit ladder demonstrates the (route, measure).

A desk bump is exact as an OPERATION on its prices, but a price with
discretisation or sampling error keeps that error after differencing. A desk
Greek is therefore ``ok`` only when every contributing price is exact (a closed
form, a fixed ledger) or Gate C demonstrated that finite move for this family
and these engine settings (review 2026-09-16 R3). Theta follows the convention:
a desk theta is the declared finite roll, a point theta a time derivative, and
each is qualified the same way as its spot counterpart (R5).
"""
from __future__ import annotations

import dataclasses
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Mapping, Optional, Sequence, Tuple

from quantark.asset.equity.engine.settlement_support import pending_receivable_pv
from quantark.asset.equity.riskmeasures.greeks import bump_envs
from quantark.execution.cache.fingerprint import fingerprint
from quantark.execution.errors import CapabilityError
from quantark.intraday.resolution import UNDER_RESOLVED
from quantark.intraday.result import GreekValue
from quantark.intraday.timestamp import SECONDS_PER_YEAR, to_utc
from quantark.util.exceptions import NumericalError, PricingError

DESK_GREEKS = ("delta", "gamma", "vega", "rho", "dividend_rho")
POINT_UNITS = {"delta": "per unit spot", "gamma": "per unit spot^2", "vega": "per unit vol (trading-quoted)",
               "rho": "per unit rate", "dividend_rho": "per unit dividend yield"}
POINT_PROXY_REASON = "finite-difference proxy for a point derivative; bump-limit not demonstrated"


@dataclass(frozen=True)
class BumpCell:
    bump_id: str                 # "base", "spot_up", "spot_down", "gamma_up", "gamma_down", "vol_up", "rate_up", "div_up"
    ctx: object                  # the resolved context with only pricing_env replaced
    price: float                 # contingent + pending receivables (re-discounted on the cell's env)
    error: str = ""
    #: The route's own resolution verdict for THIS cell's price ("resolved", "under_resolved", ...);
    #: empty when the route reports none (a closed form has nothing to resolve).
    resolution: str = ""
    resolution_reason: str = ""
    #: The route's own statement that this price carries no discretisation or sampling error.
    exact: bool = False


def bump_config_for(engine):
    from quantark.asset.equity.param import EngineParams
    params = getattr(engine, "params", None)
    getter = getattr(params, "get_effective_bump_config", None)
    return getter() if getter is not None else EngineParams().get_effective_bump_config()


def with_pricing_env(ctx, env, bump_id: str):
    """The context with only its numerical environment replaced (numerical twin and history shared by identity)."""
    child = fingerprint((ctx.market_snapshot_id, bump_id))
    return dataclasses.replace(ctx, pricing_env=env, market_snapshot_id=child)


def cell_outcome(ctx, engine):
    """(price of the whole remaining claim, route outcome) for one bump cell."""
    from quantark.intraday.engines import route_for
    state = ctx.numerical.lifecycle_state
    pending = float(pending_receivable_pv(state, ctx.pricing_env)) if state is not None else 0.0
    outcome = route_for(ctx, engine).price(ctx, engine)
    return float(outcome.contingent_pv) + pending, outcome


def cell_price(ctx, engine) -> float:
    return cell_outcome(ctx, engine)[0]


def _spot_env(env, factor: float):
    bumped = deepcopy(env)
    bumped.spot_quote.spot *= factor
    return bumped


def desk_bump_envs(ctx, engine, greeks: Sequence[str]) -> Mapping[str, object]:
    """bump id -> the environment of that desk cell (the daily builders and bump sizes, the base included)."""
    unknown = sorted(set(greeks) - set(DESK_GREEKS) - {"theta"})
    if unknown:
        raise CapabilityError(f"desk-bump greeks {unknown} are not in the intraday inventory; available {list(DESK_GREEKS)}")
    bc = bump_config_for(engine)
    env = ctx.pricing_env
    specs = {"base": env}
    if "delta" in greeks or "gamma" in greeks:
        specs["spot_up"], specs["spot_down"] = _spot_env(env, 1.0 + bc.spot_bump), _spot_env(env, 1.0 - bc.spot_bump)
    gamma_bump = bc.gamma_spot_bump if getattr(bc, "gamma_spot_bump", None) else bc.spot_bump
    if "gamma" in greeks and gamma_bump != bc.spot_bump:
        specs["gamma_up"], specs["gamma_down"] = _spot_env(env, 1.0 + gamma_bump), _spot_env(env, 1.0 - gamma_bump)
    if "vega" in greeks:
        strike = float(getattr(ctx.numerical.product, "strike", env.spot) or env.spot)
        unit_vol = bump_envs.bump_unit_vol(env, strike, max(ctx.numerical.maturity_tau, 0.0))
        specs["vol_up"] = bump_envs.build_vol_bumped_env(env, ctx.numerical.product, unit_vol, bc.vol_bump, direction=1.0)
    if "rho" in greeks:
        specs["rate_up"] = bump_envs.build_rate_bumped_env(env, bc.rate_bump, direction=1.0)
    if "dividend_rho" in greeks:
        specs["div_up"] = bump_envs.build_div_bumped_env(env, ctx.numerical.product, 0.0, bc.div_bump, direction=1.0)
    return specs


def desk_bump_cells(ctx, engine, greeks: Sequence[str]) -> Mapping[str, BumpCell]:
    cells = {}
    for bump_id, cell_env in desk_bump_envs(ctx, engine, greeks).items():
        cell_ctx = ctx if bump_id == "base" else with_pricing_env(ctx, cell_env, bump_id)
        try:
            price, outcome = cell_outcome(cell_ctx, engine)
            cells[bump_id] = BumpCell(bump_id, cell_ctx, price,
                                      resolution=str(outcome.numerical.get("resolution") or ""),
                                      resolution_reason=str(outcome.numerical.get("resolution_reason") or ""),
                                      exact=bool(outcome.exact))
        except (NumericalError, PricingError) as exc:
            cells[bump_id] = BumpCell(bump_id, cell_ctx, float("nan"), f"{type(exc).__name__}: {exc}")
    return cells


def discontinuity_at_spot(ctx, spot: Optional[float] = None) -> str:
    """Why the SURVIVING price function jumps or kinks at the query spot ("" when it does not).

    Only an UNRESOLVED event can make the conditional price function jump: one
    determined AT the valuation instant (phase BEFORE) whose level is the query
    spot leaves one side of every neighbourhood decided and the other not.

    An already-assumed continuous hit is not such an event. The design freezes
    every confirmed and assumed fixing across market bumps, so bumping spot does
    not re-decide it: the surviving function is the conditional one (a knocked-out
    contract's fixed ledger, a knocked-in contract's vanilla), and it is smooth.
    The uncertainty about that assumed history is disclosed by ``provisional`` and
    ``continuous_assumption`` — it is not a property of the market derivative.
    """
    if ctx.numerical.terminated:
        return ""                       # what survives is a fixed cash ledger, flat in spot
    now, spot = to_utc(ctx.valuation_timestamp), float(ctx.spot if spot is None else spot)
    for event in ctx.timeline.remaining(ctx.valuation_timestamp, ctx.phase):
        if to_utc(event.timestamp) == now and event.barrier is not None and float(event.barrier) == spot:
            return f"payoff discontinuity of the unfixed event {event.event_id} at the query spot"
    return ""


def point_greek_values(ctx, engine, greeks: Sequence[str]) -> Tuple[Tuple[GreekValue, ...], Tuple[str, ...]]:
    """(greek values, records) under the point convention, in the requested order."""
    from quantark.intraday.engines import route_for
    from quantark.intraday.engines.base import PointGreeks

    unknown = sorted(set(greeks) - set(POINT_UNITS))
    if unknown:
        raise CapabilityError(f"point greeks {unknown} are not in the intraday inventory; available {list(POINT_UNITS)}")
    route = route_for(ctx, engine)
    values, records = {}, []
    if "delta" in greeks or "gamma" in greeks:
        jump = discontinuity_at_spot(ctx)
        if jump:
            pg = PointGreeks(None, None, "undefined", jump, "")
        else:
            finder = getattr(route, "point_greeks", None)
            if finder is None:
                raise CapabilityError(f"{type(route).__name__} has no point greeks")
            pg = finder(ctx, engine)
        records.append(f"point_evidence:{pg.evidence or 'none'}")
        records.extend(f"point_uncertainty:{k}={v!r}" for k, v in pg.uncertainty.items())
        for name in ("delta", "gamma"):
            if name in greeks:
                ok = pg.status == "ok"
                values[name] = GreekValue(name, getattr(pg, name) if ok else None, POINT_UNITS[name], "point",
                                          status=pg.status, reason=None if ok else pg.reason)
    for name in ("vega", "rho", "dividend_rho"):
        if name in greeks:
            values[name] = _point_proxy(ctx, engine, route, name)
    return tuple(values[g] for g in greeks), tuple(records)


def _unit_vol(ctx) -> float:
    env = ctx.pricing_env
    strike = float(getattr(ctx.numerical.product, "strike", env.spot) or env.spot)
    return bump_envs.bump_unit_vol(env, strike, max(ctx.numerical.maturity_tau, 0.0))


#: Why a zero-weight window has no volatility sensitivity at all.
ZERO_CLOCK_VEGA_REASON = ("the trading clock does not advance before expiry: scaling a quoted volatility adds no "
                          "variance to a zero-weight window, so this contract's vega is exactly zero")


def zero_variance_clock(ctx) -> bool:
    """True when no trading time elapses between valuation and expiry.

    ``bump_unit_vol`` reads the level in the unit the bump moves (sigma_td) and
    returns exactly 0.0 only when the clock's trading time to expiry is zero — a
    lunch break, a holiday, or expiry itself. The quoted surfaces all require a
    positive volatility, so this is never a zero MARKET vol: it is a deterministic
    payoff, and the bump builders must not be asked for a stressed vol of zero.
    """
    return _unit_vol(ctx) == 0.0


def point_proxy_bump(ctx, name: str) -> float:
    """The production bump of a point proxy: 1e-4 of the trading-quoted vol (vega), 1e-6 (rho, dividend rho)."""
    return 1e-4 * _unit_vol(ctx) if name == "vega" else 1e-6


def point_proxy_env(ctx, name: str, h: float, direction: float):
    """The environment moved by ``direction * h`` in the proxy's market variable (the daily bump builders)."""
    env, product = ctx.pricing_env, ctx.numerical.product
    if name == "vega":
        return bump_envs.build_vol_bumped_env(env, product, _unit_vol(ctx), h, direction=direction)
    if name == "rho":
        return bump_envs.build_rate_bumped_env(env, h, direction=direction)
    if name == "dividend_rho":
        return bump_envs.build_div_bumped_env(env, product, 0.0, h, direction=direction)
    raise CapabilityError(f"{name} has no point proxy")


def point_proxy_difference(ctx, name: str, h: float, price) -> float:
    """Central difference of ``price(context)`` over the proxy's market variable with bump ``h``."""
    up = price(with_pricing_env(ctx, point_proxy_env(ctx, name, h, 1.0), f"point_{name}_up:{h!r}"))
    down = price(with_pricing_env(ctx, point_proxy_env(ctx, name, h, -1.0), f"point_{name}_down:{h!r}"))
    return (up - down) / (2.0 * h)


def _point_proxy(ctx, engine, route, name: str) -> GreekValue:
    """Central difference of the frozen price function, ``ok`` only inside a Gate C demonstrated bump limit."""
    if name == "vega" and zero_variance_clock(ctx):
        return GreekValue(name, 0.0, POINT_UNITS[name], "point", bump=0.0, reason=ZERO_CLOCK_VEGA_REASON)
    h = point_proxy_bump(ctx, name)
    if ctx.numerical.terminated:
        # No contingent claim survives: the remaining value is a FIXED cash ledger, whose
        # market derivatives are exact, not bump-limit proxies. Vol and dividends do not
        # enter it; a pending settlement's discount factors do, so rho is NOT zero.
        return GreekValue(name, point_proxy_difference(ctx, name, h, lambda c: cell_price(c, engine)),
                          POINT_UNITS[name], "point", bump=h, reason="exact derivative of the remaining fixed ledger")
    gap = certificate_gap(ctx, engine, route, f"point_{name}")
    if gap:
        return GreekValue(name, None, POINT_UNITS[name], "point", bump=h, status="unqualified",
                          reason=f"{POINT_PROXY_REASON}: {gap}")
    return GreekValue(name, point_proxy_difference(ctx, name, h, lambda c: cell_price(c, engine)), POINT_UNITS[name],
                      "point", bump=h)


def qualification_scope(ctx) -> dict:
    """What a Gate C certificate must have covered to speak for THIS request.

    A demonstration earned on discrete monitoring under the desk profile says
    nothing about the same product with a continuously observed barrier, or under
    a profile whose weights put the variance somewhere else.
    """
    from quantark.intraday.events import monitoring_of
    from quantark.intraday.capability import economic_identity
    return {"monitoring": monitoring_of(ctx.timeline),
            "profile_identity": ctx.request.variance_profile.identity(),
            "economics": economic_identity(ctx)}


def certificate_gap(ctx, engine, route, measure: str, measure_settings: Optional[dict] = None, *, spot=None) -> str:
    """"" when Gate C demonstrated ``measure`` for this request's family, engine settings and horizon; else why not."""
    from quantark.intraday.capability import accuracy_settings, output_qualification_gap
    gap = output_qualification_gap(type(ctx.request.product).__name__, type(route).__name__, measure,
                                    seconds_to_first_event(ctx), **qualification_scope(ctx),
                                    settings=accuracy_settings(engine), measure_settings=measure_settings)
    if gap or type(route).__name__ not in ("QuadV2Route", "PDERoute", "MCRoute"):
        return gap
    # The swept spot envelope: two standard deviations or ten basis points
    # around the barrier levels, whichever is wider. Do not extend it to an
    # arbitrary spot merely because all other certificate fields match.
    from math import exp, sqrt
    spot = ctx.spot if spot is None else float(spot)
    levels = [float(e.barrier) for e in ctx.numerical.remaining_events if e.barrier is not None]
    if not levels:
        levels = [float(ctx.numerical.product.strike)]
    tau = seconds_to_first_event(ctx) / SECONDS_PER_YEAR
    strike = float(getattr(ctx.numerical.product, "strike", levels[0]))
    sd = sqrt(max(float(ctx.pricing_env.vol_surface.total_variance(strike, tau, spot)), 0.0))
    lo, hi = min(levels) * min(exp(-2*sd), 0.999), max(levels) * max(exp(2*sd), 1.001)
    if not lo * (1-1e-12) <= spot <= hi * (1+1e-12):
        return f"spot {spot:g} is outside the Gate C spot envelope [{lo:g}, {hi:g}]"
    return ""


def point_certificate_gap(ctx, engine, route, *, spot=None) -> str:
    """The Gate C gaps of point delta and gamma for this request, joined ("" when both are demonstrated)."""
    gaps = (certificate_gap(ctx, engine, route, f"point_{m}", spot=spot) for m in ("delta", "gamma"))
    return "; ".join(dict.fromkeys(g for g in gaps if g))


def seconds_to_first_event(ctx) -> float:
    """Seconds from the valuation instant to the first remaining event (0.0 at an event under BEFORE)."""
    now = to_utc(ctx.valuation_timestamp)
    upcoming = [to_utc(e.timestamp) for e in ctx.numerical.remaining_events]
    return min((t - now).total_seconds() for t in upcoming) if upcoming else float("inf")


@dataclass(frozen=True)
class ThetaStep:
    """The forward step a local theta takes: requested, actually taken (None when no step exists), and its unit."""

    requested: timedelta
    actual: Optional[timedelta]
    adjusted: bool
    side: str                    # "forward" | "forward_clamped_to_event" | "forward_clamped_to_clock" | "none"
    unit: str
    divisor: Optional[float]     # actual seconds / THETA_UNITS[unit]
    reason: str = ""


DEFAULT_THETA_STEP = timedelta(hours=1)

#: What a DESK theta is: exactly the frozen-market change over ``bump`` seconds, per unit of time.
THETA_ROLL_CONVENTION = ("one-sided forward roll on the frozen market, inside the current event and clock segment; "
                         "a declared finite roll, not a demonstrated time-derivative limit")
#: A POINT theta is the time derivative itself. Its stencil is second order, with a step that is this fraction of
#: the distance to the end of the current segment (the next event, clock or coefficient boundary): the value
#: function varies on that scale, so the truncation error is O(1e-6) of the derivative, and a stencil that never
#: leaves the segment never differences across a jump in the coefficients.
POINT_THETA_RELATIVE_STEP = 1e-3
#: ... and never below a millisecond: the stencil divides a price difference by 2h, so round-off in the prices
#: (~1e-16 of their size) grows as 1/h. A segment shorter than a second leaves no step that is both local and above
#: that floor, and the derivative is reported unqualified rather than as amplified round-off.
POINT_THETA_MIN_STEP = timedelta(milliseconds=1)
POINT_THETA_REASON = ("time derivative of the frozen-market price function: one-sided second-order difference "
                      "(-3 V(0) + 4 V(h) - V(2h)) / 2h inside the current event and clock segment")
#: A step longer than any contract horizon, used to find where the current segment ends.
_SEGMENT_PROBE = timedelta(days=36500)


def theta_boundaries(ctx, until):
    """Clock and coefficient instants strictly after valuation and at or before ``until``.

    A local theta is only local inside ONE piece. Besides contractual events, the
    pieces end at the variance clock's own segment boundaries (a session open, a
    lunch break, a close) and at the market curves' pillars. Segment boundaries are
    read as instants, never as round-tripped year fractions, so a step lands exactly
    on the boundary rather than a nanosecond either side of it.

    ``until`` is capped at the clock's horizon: there are no coefficients to ask
    about beyond it, and the caller has already bounded the step by the contract's
    own events, which all fall inside it.
    """
    from quantark.intraday.coefficients import coefficient_breaks
    from quantark.intraday.timestamp import SECONDS_PER_YEAR

    now = to_utc(ctx.valuation_timestamp)
    until = min(until, to_utc(ctx.time_map.horizon_date))
    clock = {to_utc(s.end) for s in ctx.time_map.segments} | {to_utc(s.start) for s in ctx.time_map.segments}
    span = max((until - now).total_seconds(), 0.0)
    breaks = coefficient_breaks(ctx, span / SECONDS_PER_YEAR) if span > 0.0 else None
    curves = set() if breaks is None else {
        now + timedelta(seconds=float(t) * SECONDS_PER_YEAR) for t in breaks.curve}
    return sorted(t for t in (clock | curves) if now < t <= until)


def resolve_theta_step(ctx, requested: Optional[timedelta], unit: str = "hour") -> ThetaStep:
    """A forward step inside the current segment, clamped to the first boundary it would cross.

    Events and coefficient/clock boundaries both end the segment: rolling a
    "local" theta across a session open or a lunch break reports the jump in the
    variance rate as time decay of the current piece. The next event is found
    first, because it bounds the window in which a coefficient could change.
    """
    from quantark.intraday.request import THETA_UNITS

    requested = DEFAULT_THETA_STEP if requested is None else requested
    now = to_utc(ctx.valuation_timestamp)
    upcoming = sorted(to_utc(e.timestamp) for e in ctx.numerical.remaining_events)
    if upcoming and upcoming[0] == now:
        return ThetaStep(requested, None, True, "none", unit, None,
                         "valuation is at an event boundary; local theta needs a one-sided step inside the segment")
    target = now + requested
    following = [t for t in upcoming if t > now]
    first_event = following[0] if following else None
    bounded = min(target, first_event) if first_event is not None else target
    boundaries = theta_boundaries(ctx, bounded)
    candidates = [t for t in (first_event, boundaries[0] if boundaries else None) if t is not None]
    stop = min(candidates) if candidates else None
    if stop is not None and stop < target:
        actual = stop - now
        # a fixing at the close is both; the contractual event is the stronger statement
        at_event = first_event is not None and stop == first_event
        side = "forward_clamped_to_event" if at_event else "forward_clamped_to_clock"
        reason = ("the requested step crosses a contractual event" if at_event
                  else "the requested step crosses a variance-clock or coefficient boundary")
        return ThetaStep(requested, actual, True, side, unit, actual.total_seconds() / THETA_UNITS[unit], reason)
    return ThetaStep(requested, requested, False, "forward", unit, requested.total_seconds() / THETA_UNITS[unit])


def _rolled_value(ctx, engine, when: timedelta):
    """(frozen-market value after rolling ``when`` forward, plus the cash paid on the way; the roll's exactness)."""
    from quantark.intraday.roll import roll_context

    rolled = roll_context(ctx, (to_utc(ctx.valuation_timestamp) + when).astimezone(ctx.valuation_timestamp.tzinfo))
    received = float(rolled.numerical.paid_cash) - float(ctx.numerical.paid_cash)
    price, outcome = cell_outcome(rolled, engine)
    return price + received, bool(outcome.exact)


def point_theta_step(ctx) -> Tuple[Optional[timedelta], str]:
    """(the point-theta stencil step h in whole microseconds, "") or (None, why no step is admissible)."""
    end = resolve_theta_step(ctx, _SEGMENT_PROBE)
    if end.actual is None:
        return None, end.reason
    span = end.actual if end.adjusted else DEFAULT_THETA_STEP       # no boundary ahead: a fixed ledger
    h = timedelta(microseconds=round(span / timedelta(microseconds=1) * POINT_THETA_RELATIVE_STEP))
    if h < POINT_THETA_MIN_STEP:
        return None, (f"the current segment ends {span.total_seconds():g} s away: a local stencil step would be below "
                      f"{POINT_THETA_MIN_STEP.total_seconds() * 1e3:g} ms, where price round-off dominates the derivative")
    return h, ""


def point_theta_estimate(ctx, engine, *, price_base: float, unit: str = "hour"):
    """(dV/dt per ``unit``, step h, every stencil price exact) of the frozen-market price function, or None.

    The same stencil serves the runtime and the Gate C ladder, so a certificate speaks for exactly this estimator.
    """
    from quantark.intraday.request import THETA_UNITS

    h, _why = point_theta_step(ctx)
    if h is None:
        return None
    v1, exact1 = _rolled_value(ctx, engine, h)
    v2, exact2 = _rolled_value(ctx, engine, 2 * h)
    seconds = h.total_seconds()
    value = (-3.0 * price_base + 4.0 * v1 - v2) / (2.0 * seconds) * THETA_UNITS[unit]
    return value, h, exact1 and exact2


def analytical_theta_limit(ctx, engine, *, price_base: float, unit: str) -> GreekValue:
    """A per-request derivative limit on admitted exact analytical prices.

    Three second-order forward stencils share five price evaluations. Their
    Richardson differences estimate truncation; an explicit cancellation floor
    covers floating-point price noise. All points remain inside one coefficient
    segment. The frozen Gate C theta budget is assessed in PnL per hour before
    conversion to the requested unit. No sampled family certificate is needed.
    """
    from math import isfinite
    from sys import float_info
    from quantark.intraday.request import THETA_UNITS

    h, why = point_theta_step(ctx)
    if h is None:
        return GreekValue("theta", None, f"PnL per {unit}", "point", status="unqualified", reason=why)
    # Exact dyadic steps in datetime's microsecond representation.
    h = timedelta(microseconds=4 * int(h / timedelta(microseconds=4)))
    prices = {0: price_base}
    for k in (1, 2, 4, 8):
        value, exact = _rolled_value(ctx, engine, k * (h / 4))
        if not exact:
            return GreekValue("theta", None, f"PnL per {unit}", "point", status="unqualified",
                              reason="the analytical theta limit requires every rolled price to be exact")
        prices[k] = value
    seconds = h.total_seconds()
    derivatives = [(-3 * prices[0] + 4 * prices[k] - prices[2*k]) / (2 * seconds * k / 4) * 3600
                   for k in (4, 2, 1)]
    coarse, mid, fine = derivatives
    rich_coarse, rich_fine = (4 * mid - coarse) / 3, (4 * fine - mid) / 3
    product = ctx.request.product
    scale = abs(float(getattr(product, "initial_price", getattr(product, "strike", 1.0))))
    scale *= abs(float(getattr(product, "contract_multiplier", 1.0)))
    # Same frozen normalised budget as Gate C: 1e-6 N or 1e-4 |theta/hour|.
    budget = max(1e-6 * scale, 1e-4 * abs(rich_fine))
    noise = 64 * float_info.epsilon * max(scale, *(abs(p) for p in prices.values())) * 3600 / (seconds / 4)
    error = max(abs(rich_fine - rich_coarse), abs(fine - mid) / 3) + noise
    converging = abs(fine - mid) <= max(abs(mid - coarse), noise)
    ok = all(isfinite(x) for x in derivatives) and converging and error <= budget
    factor = THETA_UNITS[unit] / 3600
    reason = ("per-request analytical theta limit: second-order steps h, h/2, h/4 inside one clock/coefficient "
              "segment; Richardson truncation estimate plus floating-point cancellation floor")
    if not ok:
        reason += "; refinement does not establish the frozen theta error budget"
    return GreekValue("theta", rich_fine * factor if ok else None, f"PnL per {unit}", "point",
                      bump=seconds / 4, status="ok" if ok else "unqualified", reason=reason,
                      error_estimate=error * factor if isfinite(error) else None,
                      error_budget=budget * factor if isfinite(budget) else None)


def intraday_theta(ctx, engine, route, step: ThetaStep, *, price_base: float, base_exact: bool,
                   convention: str) -> GreekValue:
    """Theta under the request's convention: the declared finite roll (desk) or the time derivative (point).

    A finite roll published under the point convention would claim a derivative it is not: an 1800 s roll of a
    digital an hour before expiry is 60.7% off its derivative (review 2026-09-16 R5). So the two are qualified as
    their spot counterparts are. A desk roll of exact prices is exact as a move; any other roll needs Gate C to have
    demonstrated the same roll (family, engine settings, requested step). A point theta is a stencil, whose
    truncation error no exactness of the prices removes, so -- like a point proxy -- it needs the certificate
    everywhere except on a fixed ledger.
    """
    unit = f"PnL per {step.unit}"
    if step.actual is None:
        return GreekValue("theta", None, unit, convention, status="undefined", reason=step.reason)
    if convention == "point":
        from quantark.intraday.engines.analytical_barrier import AnalyticalBarrierRoute
        if isinstance(route, AnalyticalBarrierRoute) and base_exact:
            return analytical_theta_limit(ctx, engine, price_base=price_base, unit=step.unit)
        estimate = point_theta_estimate(ctx, engine, price_base=price_base, unit=step.unit)
        if estimate is None:
            return GreekValue("theta", None, unit, convention, status="unqualified", reason=point_theta_step(ctx)[1])
        value, h, _exact = estimate
        disclosure, measure, knobs = POINT_THETA_REASON, "point_theta", None
        bump = h.total_seconds()
        needs_certificate, why = not ctx.numerical.terminated, "a stencil's truncation needs a demonstrated limit"
    else:
        rolled, exact = _rolled_value(ctx, engine, step.actual)
        value = (rolled - price_base) / step.divisor
        disclosure = f"{THETA_ROLL_CONVENTION}{'; ' + step.reason if step.reason else ''}"
        measure, knobs = "desk_theta", {"theta_step_s": step.requested.total_seconds()}
        bump = step.actual.total_seconds()
        needs_certificate, why = not (exact and base_exact), "the prices carry discretisation or sampling error"
    if needs_certificate:
        gap = certificate_gap(ctx, engine, route, measure, knobs)
        if gap:
            return GreekValue("theta", None, unit, convention, bump=bump, status="unqualified",
                              reason=f"{disclosure}; {why}: {gap}")
    return GreekValue("theta", value, unit, convention, bump=bump, reason=disclosure)


def theta_metadata(step: ThetaStep) -> dict:
    return {"theta_step_requested_s": step.requested.total_seconds(),
            "theta_step_actual_s": None if step.actual is None else step.actual.total_seconds(),
            "theta_adjusted": step.adjusted, "theta_side": step.side, "theta_unit": step.unit}


def assemble_desk_greeks(cells: Mapping[str, BumpCell], greeks: Sequence[str], *, spot: float, bump_config,
                         certificate: Callable[[str], str]) -> Tuple[GreekValue, ...]:
    """Desk Greeks from the repriced cells; ``certificate(measure)`` is the Gate C gap of ``desk_<name>`` ("" = demonstrated)."""
    bc = bump_config
    out = []

    def failed(name, unit, bump, *ids):
        errors = [cells[i].error for i in ids if cells[i].error]
        if errors:
            return GreekValue(name, None, unit, "desk_bump", bump=bump, status="failed", reason="; ".join(errors))
        # A desk bump IS its repriced difference, so its only error is the prices' own.
        # A cell the route says it could NOT resolve therefore makes the Greek unqualified
        # rather than ok (review 2026-09-16 finding 4). Only that verdict is a failure to
        # deliver: "deterministic" and "not_solved" are exact prices with no diffusion to
        # resolve, "sampling_uncertainty_reported" carries its own error estimate, and a
        # closed form reports none at all.
        unresolved = {cells[i].resolution_reason or UNDER_RESOLVED
                      for i in ids if cells[i].resolution == UNDER_RESOLVED}
        if unresolved:
            return GreekValue(name, None, unit, "desk_bump", bump=bump, status="unqualified",
                              reason="a bump cell priced on a grid the route could not resolve: "
                                     + "; ".join(sorted(unresolved)))
        # Resolution is necessary, not sufficient: a resolved 418-point mesh still published a desk gamma 35x
        # its budget off the reference (review 2026-09-16 R3). Exact prices difference exactly; any other move
        # needs a demonstration of the same finite move at the same settings.
        if not all(cells[i].exact for i in ids):
            gap = certificate(f"desk_{name}")
            if gap:
                return GreekValue(name, None, unit, "desk_bump", bump=bump, status="unqualified",
                                  reason=f"a desk move differences prices that carry discretisation or sampling "
                                         f"error: {gap}")
        return None

    base = cells["base"].price
    for name in greeks:
        if name == "delta":
            h = bc.spot_bump
            unit, bump = "per unit spot", spot * h
            out.append(failed(name, unit, bump, "base", "spot_up", "spot_down") or GreekValue(
                name, (cells["spot_up"].price - cells["spot_down"].price) / (2.0 * spot * h), unit, "desk_bump", bump=bump))
        elif name == "gamma":
            h = bc.gamma_spot_bump if getattr(bc, "gamma_spot_bump", None) else bc.spot_bump
            up, down = ("gamma_up", "gamma_down") if "gamma_up" in cells else ("spot_up", "spot_down")
            unit, bump = "per unit spot^2", spot * h
            out.append(failed(name, unit, bump, "base", up, down) or GreekValue(
                name, (cells[up].price - 2.0 * base + cells[down].price) / (spot * h) ** 2, unit, "desk_bump", bump=bump))
        elif name == "vega":
            unit = f"PnL per +{bc.vol_bump:g} vol (raw, trading-quoted)"
            out.append(failed(name, unit, bc.vol_bump, "base", "vol_up") or GreekValue(
                name, cells["vol_up"].price - base, unit, "desk_bump", bump=bc.vol_bump))
        elif name == "rho":
            unit = "PnL per +1% rate"
            out.append(failed(name, unit, bc.rate_bump, "base", "rate_up") or GreekValue(
                name, (cells["rate_up"].price - base) * (0.01 / bc.rate_bump), unit, "desk_bump", bump=bc.rate_bump))
        elif name == "dividend_rho":
            unit = "PnL per +1% dividend yield"
            out.append(failed(name, unit, bc.div_bump, "base", "div_up") or GreekValue(
                name, (cells["div_up"].price - base) * (0.01 / bc.div_bump), unit, "desk_bump", bump=bc.div_bump))
    return tuple(out)
