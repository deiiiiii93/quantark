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
from typing import Mapping, Sequence, Tuple

from quantark.asset.equity.engine.settlement_support import pending_receivable_pv
from quantark.asset.equity.riskmeasures.greeks import bump_envs
from quantark.execution.cache.fingerprint import fingerprint
from quantark.execution.errors import CapabilityError
from quantark.intraday.result import GreekValue
from quantark.intraday.timestamp import same_instant, to_utc
from quantark.util.exceptions import NumericalError, PricingError

DESK_GREEKS = ("delta", "gamma", "vega", "rho", "dividend_rho")
POINT_UNITS = {"delta": "per unit spot", "gamma": "per unit spot^2", "vega": "per unit vol (trading-quoted)",
               "rho": "per unit rate", "dividend_rho": "per unit dividend yield"}
POINT_PROXY_REASON = "finite-difference proxy for a point derivative; bump-limit not demonstrated"
#: (route class name, measure) pairs whose central-difference proxy a Gate C bump ladder has demonstrated.
POINT_PROXY_DEMONSTRATED: frozenset = frozenset()


@dataclass(frozen=True)
class BumpCell:
    bump_id: str                 # "base", "spot_up", "spot_down", "gamma_up", "gamma_down", "vol_up", "rate_up", "div_up"
    ctx: object                  # the resolved context with only pricing_env replaced
    price: float                 # contingent + pending receivables (re-discounted on the cell's env)
    error: str = ""


def bump_config_for(engine):
    from quantark.asset.equity.param import EngineParams
    params = getattr(engine, "params", None)
    getter = getattr(params, "get_effective_bump_config", None)
    return getter() if getter is not None else EngineParams().get_effective_bump_config()


def with_pricing_env(ctx, env, bump_id: str):
    """The context with only its numerical environment replaced (numerical twin and history shared by identity)."""
    child = fingerprint((ctx.market_snapshot_id, bump_id))
    return dataclasses.replace(ctx, pricing_env=env, market_snapshot_id=child)


def cell_price(ctx, engine) -> float:
    from quantark.intraday.engines import route_for
    state = ctx.numerical.lifecycle_state
    pending = float(pending_receivable_pv(state, ctx.pricing_env)) if state is not None else 0.0
    return float(route_for(ctx, engine).price(ctx, engine).contingent_pv) + pending


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
            cells[bump_id] = BumpCell(bump_id, cell_ctx, cell_price(cell_ctx, engine))
        except (NumericalError, PricingError) as exc:
            cells[bump_id] = BumpCell(bump_id, cell_ctx, float("nan"), f"{type(exc).__name__}: {exc}")
    return cells


def discontinuity_at_spot(ctx) -> str:
    """Why the price function jumps or kinks at the query spot ("" when it does not).

    An event determined AT the valuation instant (phase BEFORE) whose level is the spot, or a continuous barrier
    the spot hits at this instant under the contract's inclusive rule: one side of any neighbourhood is decided,
    the other is not.
    """
    now, spot = to_utc(ctx.valuation_timestamp), float(ctx.spot)
    for event in ctx.timeline.remaining(ctx.valuation_timestamp, ctx.phase):
        if to_utc(event.timestamp) == now and event.barrier is not None and float(event.barrier) == spot:
            return f"payoff discontinuity of the unfixed event {event.event_id} at the query spot"
    assumption = ctx.reconstruction.continuous_assumption
    barrier = ctx.timeline.continuous_barrier
    level = barrier.level if barrier is not None else ctx.timeline.continuous_ki_barrier
    if (assumption is not None and assumption.assumed_hit_at is not None and level is not None
            and same_instant(assumption.assumed_hit_at, ctx.valuation_timestamp) and float(level) == spot):
        return f"the continuous barrier {level:g} is hit exactly at the query spot"
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


def _point_proxy(ctx, engine, route, name: str) -> GreekValue:
    """Central difference of the frozen price function: h = 1e-4 sigma (vega), 1e-6 (rho, dividend rho)."""
    env, product = ctx.pricing_env, ctx.numerical.product
    if name == "vega":
        strike = float(getattr(product, "strike", env.spot) or env.spot)
        unit_vol = bump_envs.bump_unit_vol(env, strike, max(ctx.numerical.maturity_tau, 0.0))
        h = 1e-4 * unit_vol
    else:
        h = 1e-6
    if (type(route).__name__, name) not in POINT_PROXY_DEMONSTRATED:
        return GreekValue(name, None, POINT_UNITS[name], "point", bump=h, status="unqualified", reason=POINT_PROXY_REASON)
    if ctx.numerical.terminated:
        return GreekValue(name, 0.0, POINT_UNITS[name], "point", bump=h)

    def env_at(direction: float):
        if name == "vega":
            return bump_envs.build_vol_bumped_env(env, product, unit_vol, h, direction=direction)
        if name == "rho":
            return bump_envs.build_rate_bumped_env(env, h, direction=direction)
        return bump_envs.build_div_bumped_env(env, product, 0.0, h, direction=direction)

    up = cell_price(with_pricing_env(ctx, env_at(1.0), f"point_{name}_up"), engine)
    down = cell_price(with_pricing_env(ctx, env_at(-1.0), f"point_{name}_down"), engine)
    return GreekValue(name, (up - down) / (2.0 * h), POINT_UNITS[name], "point", bump=h)


def assemble_desk_greeks(cells: Mapping[str, BumpCell], greeks: Sequence[str], *, spot: float, bump_config) -> Tuple[GreekValue, ...]:
    bc = bump_config
    out = []

    def failed(name, unit, bump, *ids):
        errors = [cells[i].error for i in ids if cells[i].error]
        return GreekValue(name, None, unit, "desk_bump", bump=bump, status="failed", reason="; ".join(errors)) if errors else None

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
