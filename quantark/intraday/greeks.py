"""Intraday Greeks: re-evaluations of the SAME resolved price function.

A bump cell replaces only the numerical environment of the resolved context;
the twin, the time-based ledger and every confirmed or assumed fixing are
shared by identity, so a bumped spot never re-decides an observation. Desk
bumps reuse the daily helpers and conventions exactly (``bump_envs`` and
``GreeksCalculator``): relative central spot bumps, a one-sided raw vega per
``vol_bump`` of the trading-quoted surface, one-sided rho and dividend rho
rescaled to +1%.
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
from quantark.util.exceptions import NumericalError, PricingError

DESK_GREEKS = ("delta", "gamma", "vega", "rho", "dividend_rho")


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
