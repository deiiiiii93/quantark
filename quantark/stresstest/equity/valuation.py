"""Per-position trade valuation for static equity stress.

Shared by the scalar and native-batch paths and by the process-pool workers. An ``EquityPosition`` is valued at
its full trade value (product × signed quantity + absolute cash legs, ``get_trade_value``) and its Greeks come from
``get_trade_risk(...).total``; a position object without the trade API (``EquitySwapPosition``) keeps
``get_market_value`` / ``get_greeks`` (ship spec 2026-10-23 D1–D4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from quantark.asset.equity.riskmeasures import GreeksCalculator

#: the six trade Greeks, bump-repriced with the calculator's bump configuration
TRADE_GREEKS = ("delta", "gamma", "vega", "theta", "rho", "dividend_rho")
#: a position's Greeks dict, in order: the historical key set (theta components estimated from the totals)
POSITION_GREEK_KEYS = ("market_value",) + TRADE_GREEKS + ("convexity_theta", "r_theta", "q_theta")
#: portfolio / per-underlying aggregates: the historical keys plus dividend_rho
AGGREGATE_GREEK_KEYS = ("market_value",) + TRADE_GREEKS


@dataclass(frozen=True)
class PositionValuation:
    """One position under one environment: its value and, when requested, its Greeks dict."""

    value: float
    greeks: Optional[Dict[str, float]] = None


def has_trade_api(position: Any) -> bool:
    """True for a position valued on the trade path (``get_trade_value`` / ``get_trade_risk``)."""
    return callable(getattr(position, "get_trade_risk", None))


def trade_greeks(position: Any, env: Any, greeks_calculator: GreeksCalculator, total: Dict[str, float]) -> Dict[str, float]:
    """A position's Greeks dict from its trade-risk ``total``: market value, the six Greeks, and the theta
    components the calculator's "estimate" mode derives from the totals."""
    T = position.product.get_maturity(env)
    components = greeks_calculator.estimate_theta_components(
        total["theta"], total["rho"], total["dividend_rho"], env.get_rate(T), env.get_div_yield(T), T
    )
    greeks = {"market_value": total["price"]}
    greeks.update((name, total[name]) for name in TRADE_GREEKS)
    greeks.update((name, components[name]) for name in ("convexity_theta", "r_theta", "q_theta"))
    return greeks


def value_position(
    position: Any,
    env: Any,
    greeks_calculator: Optional[GreeksCalculator] = None,
    *,
    use_analytical: bool = True,
) -> PositionValuation:
    """Value one position under ``env``; Greeks only when a calculator is given.

    ``use_analytical`` applies to positions without the trade API only (``greeks_method``, D4).
    """
    if not has_trade_api(position):
        value = position.get_market_value(env)
        greeks = None
        if greeks_calculator is not None:
            greeks = position.get_greeks(env, greeks_calculator, use_analytical=use_analytical)
        return PositionValuation(value, greeks)
    value = position.get_trade_value(env)
    if greeks_calculator is None:
        return PositionValuation(value)
    total = position.get_trade_risk(env, greeks_calculator).total
    return PositionValuation(value, trade_greeks(position, env, greeks_calculator, total))


def aggregate_greeks(greeks_dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Sum position Greeks over ``AGGREGATE_GREEK_KEYS``; a key a position does not report adds nothing."""
    aggregated = {key: 0.0 for key in AGGREGATE_GREEK_KEYS}
    for greeks in greeks_dicts:
        for key in aggregated:
            if key in greeks:
                aggregated[key] += greeks[key]
    return aggregated
