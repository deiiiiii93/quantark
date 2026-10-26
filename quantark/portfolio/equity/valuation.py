"""Per-position valuation switch shared by every equity consumer.

One place decides how a position is valued, so the portfolio, VaR, SIMM, static stress, dynamic scenarios and
backtests cannot drift apart:

- trade mode (``EquityPortfolio.trade_value``): an ``EquityPosition`` is valued at its full trade value (product
  × signed quantity + absolute cash legs, ``get_trade_value``), its Greeks come from ``get_trade_risk(...).total``
  (bump Greeks for every position) and its PnL from ``get_trade_pnl``;
- product mode: ``get_market_value`` / ``get_greeks(use_analytical=...)`` / ``get_pnl``, as before.

A position object without the trade API (``EquitySwapPosition``) keeps ``get_market_value`` / ``get_greeks`` /
``get_pnl`` in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from quantark.asset.equity.riskmeasures import GreeksCalculator

#: the six trade Greeks, bump-repriced with the calculator's bump configuration
TRADE_GREEKS = ("delta", "gamma", "vega", "theta", "rho", "dividend_rho")
#: a position's trade Greeks dict, in order: the historical key set (theta components estimated from the totals)
POSITION_GREEK_KEYS = ("market_value",) + TRADE_GREEKS + ("convexity_theta", "r_theta", "q_theta")
#: trade-mode portfolio / per-underlying aggregates: the historical keys plus dividend_rho
AGGREGATE_GREEK_KEYS = ("market_value",) + TRADE_GREEKS
#: product-mode portfolio / per-underlying aggregates (the historical key set)
PRODUCT_AGGREGATE_KEYS = ("market_value", "delta", "gamma", "vega", "theta", "rho")


@dataclass(frozen=True)
class PositionValuation:
    """One position under one environment: its value and, when requested, its Greeks dict."""

    value: float
    greeks: Optional[Dict[str, float]] = None


def has_trade_api(position: Any) -> bool:
    """True for a position that can be valued on the trade path (``get_trade_value`` / ``get_trade_risk``)."""
    return callable(getattr(position, "get_trade_risk", None))


def supports_native_batch(position: Any) -> bool:
    """True when the position's engine offers a spot-batch session (QUAD V2) and the position the trade API."""
    return has_trade_api(position) and bool(getattr(position.engine, "supports_spot_batch", False))


def trade_greeks(
    position: Any,
    env: Any,
    greeks_calculator: GreeksCalculator,
    total: Dict[str, float],
    *,
    market_value: Optional[float] = None,
) -> Dict[str, float]:
    """A position's Greeks dict from its trade-risk ``total``: market value, the six Greeks, and the theta
    components the calculator's "estimate" mode derives from the totals.

    ``market_value`` is the value the caller reports for the position (the value path); without it the
    trade-risk base ``total["price"]`` is used, as on the batch path where both are the same solve.
    """
    T = position.product.get_maturity(env)
    components = greeks_calculator.estimate_theta_components(
        total["theta"], total["rho"], total["dividend_rho"], env.get_rate(T), env.get_div_yield(T), T
    )
    greeks = {"market_value": total["price"] if market_value is None else market_value}
    greeks.update((name, total[name]) for name in TRADE_GREEKS)
    greeks.update((name, components[name]) for name in ("convexity_theta", "r_theta", "q_theta"))
    return greeks


def value_position(
    position: Any,
    env: Any,
    greeks_calculator: Optional[GreeksCalculator] = None,
    *,
    trade: bool,
    use_analytical: bool = True,
) -> PositionValuation:
    """Value one position under ``env``; Greeks only when a calculator is given.

    ``use_analytical`` applies to the product path only (product mode, or a position without the trade API).
    """
    if not (trade and has_trade_api(position)):
        value = position.get_market_value(env)
        greeks = None
        if greeks_calculator is not None:
            greeks = position.get_greeks(env, greeks_calculator, use_analytical=use_analytical)
        return PositionValuation(value, greeks)
    value = position.get_trade_value(env)
    if greeks_calculator is None:
        return PositionValuation(value)
    total = position.get_trade_risk(env, greeks_calculator).total
    return PositionValuation(value, trade_greeks(position, env, greeks_calculator, total, market_value=value))


def position_pnl(position: Any, env: Any, *, trade: bool) -> float:
    """A position's unrealized PnL: ``get_trade_pnl`` in trade mode (trade API only), else ``get_pnl``."""
    if trade and has_trade_api(position):
        return position.get_trade_pnl(env)
    return position.get_pnl(env)


def value_positions(
    positions: Dict[str, Any],
    envs: Dict[str, Any],
    greeks_calculator: Optional[GreeksCalculator] = None,
    *,
    trade: bool,
    use_analytical: bool = True,
) -> Dict[str, PositionValuation]:
    """Each position valued once under its underlying's environment, in the positions' order."""
    return {
        position_id: value_position(
            position, envs[position.underlying], greeks_calculator, trade=trade, use_analytical=use_analytical
        )
        for position_id, position in positions.items()
    }


def batch_value_position(
    position: Any,
    envs: Sequence[Any],
    greeks_calculator: Optional[GreeksCalculator] = None,
) -> List[PositionValuation]:
    """A session-capable position under every environment, in order, in one spot-batch session (trade mode).

    With Greeks the value is the trade-risk base (``total["price"]``, the same solve the bumps start from);
    without, ``batch_trade_values``. The caller passes a copy it owns: the engine's session state lives on the
    position's engine for the duration of the call.
    """
    if greeks_calculator is None:
        return [PositionValuation(value) for value in position.batch_trade_values(envs)]
    risks = position.batch_trade_risk(envs, greeks_calculator)
    return [
        PositionValuation(risk.total["price"], trade_greeks(position, env, greeks_calculator, risk.total))
        for env, risk in zip(envs, risks)
    ]


def aggregate_greeks(
    greeks_dicts: Iterable[Dict[str, float]], keys: Sequence[str] = AGGREGATE_GREEK_KEYS
) -> Dict[str, float]:
    """Sum position Greeks over ``keys``; a key a position does not report adds nothing."""
    aggregated = {key: 0.0 for key in keys}
    for greeks in greeks_dicts:
        for key in aggregated:
            if key in greeks:
                aggregated[key] += greeks[key]
    return aggregated
