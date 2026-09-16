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
"""
from __future__ import annotations

import dataclasses
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from typing import Mapping, Optional, Sequence, Tuple

from quantark.asset.equity.engine.settlement_support import pending_receivable_pv
from quantark.asset.equity.riskmeasures.greeks import bump_envs
from quantark.execution.cache.fingerprint import fingerprint
from quantark.execution.errors import CapabilityError
from quantark.intraday.result import GreekValue
from quantark.intraday.timestamp import to_utc
from quantark.util.exceptions import NumericalError, PricingError

DESK_GREEKS = ("delta", "gamma", "vega", "rho", "dividend_rho")
POINT_UNITS = {"delta": "per unit spot", "gamma": "per unit spot^2", "vega": "per unit vol (trading-quoted)",
               "rho": "per unit rate", "dividend_rho": "per unit dividend yield"}
POINT_PROXY_REASON = "finite-difference proxy for a point derivative; bump-limit not demonstrated"
#: The one route resolution verdict that means "this price is not the one you asked for".
UNRESOLVED = "unqualified"


@dataclass(frozen=True)
class BumpCell:
    bump_id: str                 # "base", "spot_up", "spot_down", "gamma_up", "gamma_down", "vol_up", "rate_up", "div_up"
    ctx: object                  # the resolved context with only pricing_env replaced
    price: float                 # contingent + pending receivables (re-discounted on the cell's env)
    error: str = ""
    #: The route's own resolution verdict for THIS cell's price ("resolved", "unqualified", ...);
    #: empty when the route reports none (a closed form has nothing to resolve).
    resolution: str = ""
    resolution_reason: str = ""


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


def desk_bump_cells(ctx, engine, greeks: Sequence[str]) -> Mapping[str, BumpCell]:
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
    cells = {}
    for bump_id, cell_env in specs.items():
        cell_ctx = ctx if bump_id == "base" else with_pricing_env(ctx, cell_env, bump_id)
        try:
            price, outcome = cell_outcome(cell_ctx, engine)
            cells[bump_id] = BumpCell(bump_id, cell_ctx, price,
                                      resolution=str(outcome.numerical.get("resolution") or ""),
                                      resolution_reason=str(outcome.numerical.get("resolution_reason") or ""))
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
    from quantark.intraday.capability import output_qualification_gap
    if name == "vega" and zero_variance_clock(ctx):
        return GreekValue(name, 0.0, POINT_UNITS[name], "point", bump=0.0, reason=ZERO_CLOCK_VEGA_REASON)
    h = point_proxy_bump(ctx, name)
    if ctx.numerical.terminated:
        # No contingent claim survives: the remaining value is a FIXED cash ledger, whose
        # market derivatives are exact, not bump-limit proxies. Vol and dividends do not
        # enter it; a pending settlement's discount factors do, so rho is NOT zero.
        return GreekValue(name, point_proxy_difference(ctx, name, h, lambda c: cell_price(c, engine)),
                          POINT_UNITS[name], "point", bump=h, reason="exact derivative of the remaining fixed ledger")
    gap = output_qualification_gap(type(ctx.request.product).__name__, type(route).__name__, f"point_{name}",
                                   seconds_to_first_event(ctx), **qualification_scope(ctx))
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
    return {"monitoring": monitoring_of(ctx.timeline),
            "profile_identity": ctx.request.variance_profile.identity()}


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

#: What a local theta IS. Saying it on every value keeps a finite roll from reading as a
#: demonstrated dV/dt: the number is exactly the frozen-market change over ``bump`` seconds.
THETA_ROLL_CONVENTION = ("one-sided forward roll on the frozen market, inside the current event and clock segment; "
                         "a declared finite roll, not a demonstrated time-derivative limit")


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


def intraday_theta(ctx, engine, step: ThetaStep, *, price_base: float, convention: str) -> GreekValue:
    """(frozen-market value after the step, with the cash it paid) - value now, per unit of time."""
    from quantark.intraday.roll import roll_context

    unit = f"PnL per {step.unit}"
    if step.actual is None:
        return GreekValue("theta", None, unit, convention, status="undefined", reason=step.reason)
    rolled = roll_context(ctx, (to_utc(ctx.valuation_timestamp) + step.actual).astimezone(ctx.valuation_timestamp.tzinfo))
    received = float(rolled.numerical.paid_cash) - float(ctx.numerical.paid_cash)
    value = (cell_price(rolled, engine) + received - price_base) / step.divisor
    disclosure = f"{THETA_ROLL_CONVENTION}{'; ' + step.reason if step.reason else ''}"
    return GreekValue("theta", value, unit, convention, bump=step.actual.total_seconds(), reason=disclosure)


def theta_metadata(step: ThetaStep) -> dict:
    return {"theta_step_requested_s": step.requested.total_seconds(),
            "theta_step_actual_s": None if step.actual is None else step.actual.total_seconds(),
            "theta_adjusted": step.adjusted, "theta_side": step.side, "theta_unit": step.unit}


def assemble_desk_greeks(cells: Mapping[str, BumpCell], greeks: Sequence[str], *, spot: float, bump_config) -> Tuple[GreekValue, ...]:
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
        unresolved = {cells[i].resolution_reason or UNRESOLVED
                      for i in ids if cells[i].resolution == UNRESOLVED}
        if unresolved:
            return GreekValue(name, None, unit, "desk_bump", bump=bump, status="unqualified",
                              reason="a bump cell priced on a grid the route could not resolve: "
                                     + "; ".join(sorted(unresolved)))
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
