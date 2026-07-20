"""Greek bumps as framework scenario cells (spec 2026-07-20).

Mirrors ``EquityPosition.get_trade_risk`` and
``GreeksCalculator.calculate_numerical_greeks`` OPERATION-FOR-OPERATION so
greeks assembled from cell values are bitwise-equal to the originals
(``test/execution/test_greek_bump_cells.py``). Any numerical difference
introduced here is a parity bug by definition — diagnose, never tolerate.

Shape: a registered factory resolves one row/trade into a ``TradeState``
(the per-plan base); the same-type transformer ``greek-bump/v1`` applies
one bump per cell (real component attribution: spot/vol/rate/div/time);
the float runner ``greek-value/v1`` performs exactly one solve per cell.
Assembly back into greek dictionaries is plain arithmetic on the parent.
"""
import dataclasses
from copy import deepcopy
from typing import Sequence

from quantark.execution.scenario import registries

__all__ = [
    "GREEK_BUMP_TRANSFORMER_ID",
    "GREEK_VALUE_RUNNER_ID",
    "GreekBumpCell",
    "TradeState",
    "apply_greek_bump",
    "assemble_product_greeks",
    "assemble_trade_greeks",
    "greek_bump_cells",
    "greek_bump_transform",
    "run_greek_bump",
]

GREEK_BUMP_TRANSFORMER_ID = "greek-bump/v1"
GREEK_VALUE_RUNNER_ID = "greek-value/v1"

_BUMP_IDS = ("base", "spot_up", "spot_down", "vol_up", "rate_up", "div_up",
             "theta")


@dataclasses.dataclass
class TradeState:
    """Resolved per-trade base the bump transformer mutates (same type in,
    same type out; never in place — the planner's purity check verifies)."""

    product: object
    pricing_env: object
    cash_legs: tuple
    engine: object
    streams: object
    quantity: float
    greeks_params: object
    warnings: tuple = ()
    theta_degenerate: bool = False


@dataclasses.dataclass(frozen=True)
class GreekBumpCell:
    bump_id: str
    greeks_served: frozenset
    mutation_tags: frozenset


_CELL_TABLE = {
    "spot_up": (frozenset({"delta", "gamma"}), frozenset({"spot"})),
    "spot_down": (frozenset({"delta", "gamma"}), frozenset({"spot"})),
    "vol_up": (frozenset({"vega"}), frozenset({"vol"})),
    "rate_up": (frozenset({"rho"}), frozenset({"rate"})),
    "div_up": (frozenset({"dividend_rho"}), frozenset({"div"})),
    "theta": (frozenset({"theta"}), frozenset({"time"})),
}


def greek_bump_cells(greeks: Sequence[str]) -> tuple:
    """Minimal cell set serving the requested greeks; base always first.

    Unknown greek names fail closed (code-gate finding 2026-07-20) —
    the calculator raises ValidationError for them, and a silently
    dropped greek would be an incomplete risk report."""
    requested = set(greeks)
    supported = {"price"}
    for served, _tags in _CELL_TABLE.values():
        supported |= served
    unknown = requested - supported
    if unknown:
        from quantark.util.exceptions import ValidationError

        raise ValidationError(
            f"Unknown greek name(s) for bump cells: {sorted(unknown)}; "
            f"supported: {sorted(supported)}"
        )
    cells = [GreekBumpCell("base", frozenset({"price"}), frozenset())]
    for bump_id in _BUMP_IDS[1:]:
        served, tags = _CELL_TABLE[bump_id]
        if served & requested:
            cells.append(GreekBumpCell(bump_id, served, tags))
    return tuple(cells)


def _greeks_calculator(state: TradeState):
    from quantark.asset.equity.riskmeasures import GreeksCalculator

    return GreeksCalculator(params=state.greeks_params)


def apply_greek_bump(bump_id: str, state: TradeState, gc) -> TradeState:
    """Same-type bump application; every mutation is the verbatim
    convention of get_trade_risk / calculate_numerical_* (see module doc)."""
    bc = gc._bump_config
    env = state.pricing_env
    product = state.product
    if bump_id == "base":
        return state
    # Linear products short-circuit BEFORE any bump logic in the
    # calculator (including theta's date validation) — mirror that: no
    # mutation is ever applied, the runner emits the linear marker.
    if not state.cash_legs and getattr(product, "is_linear", False):
        return state
    if bump_id in ("spot_up", "spot_down"):
        direction = 1.0 if bump_id == "spot_up" else -1.0
        env2 = deepcopy(env)
        env2.spot_quote.spot *= 1.0 + direction * bc.spot_bump
        return dataclasses.replace(state, pricing_env=env2)
    T = product.get_maturity(env)
    if bump_id == "vol_up":
        strike = getattr(product, "strike", env.spot)
        cur_vol = env.get_vol(strike, T)
        env2 = gc._build_vol_bumped_env(
            env, product, cur_vol, bc.vol_bump, direction=1.0
        )
        return dataclasses.replace(state, pricing_env=env2)
    if bump_id == "rate_up":
        from quantark.param.rrf import FlatRateCurve

        cur_rate = env.get_rate(T)
        env2 = deepcopy(env)
        env2.rate_curve = FlatRateCurve(cur_rate + bc.rate_bump)
        return dataclasses.replace(state, pricing_env=env2)
    if bump_id == "div_up":
        cur_div = env.get_div_yield(T)
        env2 = gc._build_div_bumped_env(
            env, product, cur_div, bc.div_bump, direction=1.0
        )
        return dataclasses.replace(state, pricing_env=env2)
    if bump_id == "theta":
        return _apply_theta_bump(state, gc)
    raise ValueError(f"unknown bump_id {bump_id!r}")


def _apply_theta_bump(state: TradeState, gc) -> TradeState:
    """Verbatim _trade_theta / calculate_numerical_theta time shift,
    including every degenerate early-out (returned as a flag so the
    assembler can emit the identical 0.0)."""
    bc = gc._bump_config
    env = state.pricing_env
    product = state.product
    bumped_date, time_bump, resolved_mode = gc._advance_theta_bump(
        env, bc.time_bump_days, getattr(bc, "time_bump_mode", "auto")
    )
    current_maturity = product.get_maturity(env)
    if time_bump <= 0.0:
        if current_maturity <= 0.0:
            return dataclasses.replace(state, theta_degenerate=True)
        if resolved_mode == "business_days":
            from quantark.util.exceptions import ValidationError

            raise ValidationError(
                "Business-day theta bump did not advance time: "
                f"valuation_date={env.valuation_date}, "
                f"bumped_date={bumped_date}, "
                f"time_bump_days={bc.time_bump_days}"
            )
        return dataclasses.replace(state, theta_degenerate=True)
    if current_maturity <= time_bump:
        return dataclasses.replace(state, theta_degenerate=True)
    prod2 = deepcopy(product)
    env2 = deepcopy(env)
    env2.valuation_date = bumped_date
    dropped_all = prod2.time_shift(time_bump, bumped_date, env2)
    if dropped_all:
        return dataclasses.replace(state, theta_degenerate=True)
    legs2 = []
    for leg in state.cash_legs:
        shift = getattr(leg, "time_shift", None)
        shifted = shift(time_bump) if callable(shift) else leg
        if shifted is not None:
            legs2.append(shifted)
    return dataclasses.replace(
        state, product=prod2, pricing_env=env2, cash_legs=tuple(legs2)
    )


def greek_bump_transform(base: TradeState, parameters: dict) -> TradeState:
    gc = _greeks_calculator(base)
    return apply_greek_bump(parameters["bump_id"], base, gc)


def _vol_summary(env):
    """Fingerprintable summary of the vol surface using the CONCRETE field
    names quantark constructs (FlatVolSurface.volatility,
    TermStructureVolSurface.vols); None for unknown types = conservative
    invalidation, never misattribution."""
    surface = env.vol_surface
    value = getattr(surface, "volatility", None)
    if value is not None:
        return float(value)
    vols = getattr(surface, "vols", None)
    if vols is not None:
        return tuple(float(v) for v in vols)
    return None


def _rate_summary(env):
    curve = env.rate_curve
    value = getattr(curve, "rate", None)
    if value is not None:
        return float(value)
    rates = getattr(curve, "rates", None)
    if rates is not None:
        return tuple(float(r) for r in rates)
    return None


def _div_summary(env):
    dy = env.div_yield
    if dy is None:
        return 0.0
    value = getattr(dy, "div_yield", None)
    if value is not None:
        return float(value)
    yields = getattr(dy, "yields", None)
    if yields is not None:
        return tuple(float(y) for y in yields)
    return None


_COMPONENTS = (
    ("spot", lambda s: float(s.pricing_env.spot)),
    ("vol", lambda s: _vol_summary(s.pricing_env)),
    ("rate", lambda s: _rate_summary(s.pricing_env)),
    ("div", lambda s: _div_summary(s.pricing_env)),
    ("time", lambda s: str(s.pricing_env.valuation_date)),
)


def _unit_notional(state: TradeState, env) -> float:
    """Mirror EquityPosition._get_unit_notional (per-solve, on the env the
    solve used — exactly as get_trade_risk's reprice())."""
    from quantark.portfolio.equity.position import EquityPosition

    pos = EquityPosition(
        product=state.product,
        quantity=state.quantity,
        entry_price=float(getattr(state.product, "initial_price", 0.0) or 0.0),
        underlying="OTC",
        engine=state.engine,
        entry_timestamp=env.valuation_date,
        cash_legs=list(state.cash_legs),
    )
    return pos._get_unit_notional(env)


def run_greek_bump(bump_id: str, base: TradeState, bumped: TradeState,
                   gc) -> dict:
    """Exactly one solve, mirroring the source loops:

    - cash-leg trades: ``price_with_events`` on the FROZEN bump context
      resolved against the BASE state (get_trade_risk §11.4) for every
      cell including base;
    - legless trades: base cell prices on the ORIGINAL engine (v023's
      ``raw_price``), bump cells on the resolved bump context
      (calculate_numerical_greeks).
    """
    from quantark.cashleg.leg_valuator import value_leg

    if bumped.theta_degenerate:
        return {"degenerate": 1.0, "npv": 0.0, "leg_sum": 0.0}
    if base.cash_legs:
        bump_engine = gc._resolve_bump_engine(
            base.product, base.pricing_env, base.engine
        )
        result = bump_engine.price_with_events(
            bumped.product, bumped.pricing_env, streams=base.streams
        )
        unit = _unit_notional(base, bumped.pricing_env)
        legs = bumped.cash_legs if bump_id == "theta" else base.cash_legs
        out = {"degenerate": 0.0, "npv": float(result.npv)}
        leg_sum = 0.0
        # Ordered (name, direction, pv) tuples — leg_id is a per-instance
        # UUID and cannot key anything across process rebuilds; v023 itself
        # aggregates leg PVs by NAME.
        per_leg = []
        for leg in legs:
            pv = float(
                value_leg(leg, result.event_distribution,
                          bumped.pricing_env, unit)
            )
            leg_sum += pv
            per_leg.append((str(leg.name), str(leg.direction), pv))
        out["leg_sum"] = leg_sum
        if bump_id == "base":
            out["per_leg"] = tuple(per_leg)
        return out
    # Linear (delta-one) products: mirror calculate_numerical_greeks'
    # is_linear short circuit (code-gate finding 2026-07-20) — bump cells
    # never solve, and assembly reproduces _greeks_for_linear.
    if getattr(base.product, "is_linear", False):
        if bump_id == "base":
            price = float(base.engine.price(base.product, base.pricing_env))
            return {"degenerate": 0.0, "npv": price, "leg_sum": 0.0,
                    "linear": 1.0}
        return {"degenerate": 0.0, "npv": 0.0, "leg_sum": 0.0, "linear": 1.0}
    if bump_id == "base":
        price = float(base.engine.price(base.product, base.pricing_env))
    else:
        bump_engine = gc._resolve_bump_engine(
            base.product, base.pricing_env, base.engine
        )
        price = float(bump_engine.price(bumped.product, bumped.pricing_env))
    return {"degenerate": 0.0, "npv": price, "leg_sum": 0.0}


def assemble_trade_greeks(base_values, bump_values, *, quantity, spot,
                          bump_config, requested) -> dict:
    """Verbatim get_trade_risk arithmetic on (npv, leg_sum) pairs."""
    bc = bump_config
    q = float(quantity)
    base_npv = base_values["npv"]
    base_legs = base_values["leg_sum"]
    product_g = {"price": q * base_npv}
    total_g = {"price": q * base_npv + base_legs}

    def record(name, d_npv, d_legs):
        product_g[name] = q * d_npv
        total_g[name] = q * d_npv + d_legs

    if {"delta", "gamma"} & requested:
        h = spot * bc.spot_bump
        nu = bump_values["spot_up"]["npv"]
        lu = bump_values["spot_up"]["leg_sum"]
        nd = bump_values["spot_down"]["npv"]
        ld = bump_values["spot_down"]["leg_sum"]
        if "delta" in requested:
            record("delta", (nu - nd) / (2.0 * h), (lu - ld) / (2.0 * h))
        if "gamma" in requested:
            record(
                "gamma",
                (nu - 2.0 * base_npv + nd) / h**2,
                (lu - 2.0 * base_legs + ld) / h**2,
            )
    if "vega" in requested:
        nv = bump_values["vol_up"]["npv"]
        lv = bump_values["vol_up"]["leg_sum"]
        record("vega", nv - base_npv, lv - base_legs)
    if "rho" in requested:
        nr = bump_values["rate_up"]["npv"]
        lr = bump_values["rate_up"]["leg_sum"]
        scale = 0.01 / bc.rate_bump
        record("rho", (nr - base_npv) * scale, (lr - base_legs) * scale)
    if "dividend_rho" in requested:
        nd_ = bump_values["div_up"]["npv"]
        ld_ = bump_values["div_up"]["leg_sum"]
        scale = 0.01 / bc.div_bump
        record("dividend_rho", (nd_ - base_npv) * scale,
               (ld_ - base_legs) * scale)
    if "theta" in requested:
        tv = bump_values["theta"]
        if tv["degenerate"]:
            product_g["theta"], total_g["theta"] = 0.0, 0.0
        else:
            npv_t, legs_t = tv["npv"], tv["leg_sum"]
            product_g["theta"] = q * (npv_t - base_npv)
            total_g["theta"] = q * (npv_t - base_npv) + (legs_t - base_legs)
    return {
        "product": product_g,
        "total": total_g,
        "leg_pvs": tuple(base_values.get("per_leg", ())),
    }


def assemble_product_greeks(*, base_price, bump_values, spot, bump_config,
                            requested) -> dict:
    """Verbatim calculate_numerical_greeks arithmetic (BUMP mode):
    central delta/gamma via _calculate_sensitivity(scale=spot), one-sided
    vega raw P&L, one-sided rho/dividend_rho with the single 0.01/bump
    rescale, theta as a difference with degenerate zero."""
    bc = bump_config
    # Linear short circuit (code-gate finding 2026-07-20): verbatim
    # _greeks_for_linear + the requested-set filtering of
    # calculate_numerical_greeks (missing keys default to 0.0).
    if any(v.get("linear") for v in bump_values.values()):
        linear = {
            "price": base_price,
            "delta": 1.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "convexity_theta": 0.0,
            "r_theta": 0.0,
            "q_theta": 0.0,
            "rho": 0.0,
            "dividend_rho": 0.0,
        }
        for extra in requested:
            linear.setdefault(extra, 0.0)
        return {key: linear[key] for key in linear if key in requested}
    out = {}
    if "price" in requested:
        out["price"] = base_price
    if {"delta", "gamma"} & requested:
        up = bump_values["spot_up"]["npv"]
        down = bump_values["spot_down"]["npv"]
        if "delta" in requested:
            out["delta"] = (up - down) / (2.0 * spot * bc.spot_bump)
        if "gamma" in requested:
            out["gamma"] = (up - 2.0 * base_price + down) / (
                spot * bc.spot_bump
            ) ** 2
    if "vega" in requested:
        out["vega"] = bump_values["vol_up"]["npv"] - base_price
    if "theta" in requested:
        tv = bump_values["theta"]
        out["theta"] = 0.0 if tv["degenerate"] else tv["npv"] - base_price
    if "rho" in requested:
        raw = bump_values["rate_up"]["npv"] - base_price
        out["rho"] = raw * (0.01 / bc.rate_bump)
    if "dividend_rho" in requested:
        raw = bump_values["div_up"]["npv"] - base_price
        out["dividend_rho"] = raw * (0.01 / bc.div_bump)
    return out


def greek_value_runner(cell, resolved, child_context):
    """value_kind='float' cell runner: one solve + deterministic manifest."""
    parameters = dict(cell.parameters)
    base = resolved.base_inputs
    bumped = resolved.transformed
    gc = _greeks_calculator(base)
    values = run_greek_bump(parameters["bump_id"], base, bumped, gc)
    economics = [
        ("npv", values["npv"]),
        ("leg_sum", values["leg_sum"]),
        ("degenerate", values["degenerate"]),
        ("linear", values.get("linear", 0.0)),
        ("spot", float(base.pricing_env.spot)),
    ]
    for i, (name, direction, pv) in enumerate(values.get("per_leg", ())):
        economics.append((f"leg_pv::{i}", pv))
        economics.append((f"leg_name::{i}", name))
        economics.append((f"leg_dir::{i}", direction))
    for i, warning in enumerate(base.warnings):
        economics.append((f"warning::{i}", str(warning)))
    from quantark.execution.cache.fingerprint import try_fingerprint
    from quantark.execution.manifest import build_versions, platform_tag

    manifest_fp = try_fingerprint(
        (GREEK_VALUE_RUNNER_ID, parameters["bump_id"], cell.cell_fingerprint,
         build_versions(), platform_tag())
    )
    return values["npv"], tuple(economics), manifest_fp


if __name__ == "quantark.execution.greeks":
    registries.register_transformer(
        GREEK_BUMP_TRANSFORMER_ID, greek_bump_transform,
        allowed_tags=frozenset({"spot", "vol", "rate", "div", "time"}),
        components=_COMPONENTS,
        covered_fields=None,
    )
    registries.register_runner(
        GREEK_VALUE_RUNNER_ID, greek_value_runner, value_kind="float"
    )
