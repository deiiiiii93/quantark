"""
Record-row schemas for the replay backtest — the single source of truth.

Every frame the replay engines emit is declared here: the TypedDict names the
fields and the ``*_COLUMNS`` tuple fixes the column ORDER. Stage-13-style
consumers treat these columns as a stable contract; changing one is a
deliberate, reviewed act (the golden gate enforces it).

Dynamic suffixes (documented, not part of the fixed tuples):

- ``StateRow`` — surface-mode runs append the provenance columns
  ``SURFACE_PROVENANCE_COLUMNS`` after ``matured``.
- ``ActionRow`` — each lifecycle event merges its ``metadata`` dict after the
  fixed prefix (e.g. ``payoff`` for KO/maturity, ``monitoring`` for
  continuous KI).
- ``CalibrationRecord`` — variant-specific fit fields follow the fixed keys
  (LV grid stats / Heston params + fit metrics / SLV leverage stats), plus
  wall-clock ``calibration_seconds`` / ``pricing_seconds`` which are excluded
  from golden comparisons.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

STATE_COLUMNS: tuple[str, ...] = (
    "date", "portfolio_value", "product_mtm", "hedge_mtm", "cash", "cashflows",
    "transaction_costs", "product_pnl", "hedge_pnl", "total_pnl", "spot",
    "volatility", "rate", "basis_yield", "implied_q", "pricing_q",
    "active_contract", "futures_price", "futures_ttm", "futures_multiplier",
    "futures_contracts", "alive", "knocked_in", "knocked_out", "matured",
)

SURFACE_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "surface_date", "surface_sha", "surface_extrapolation", "surface_max_listed_T",
)

GREEK_COLUMNS: tuple[str, ...] = (
    "date", "price", "delta", "gamma", "product_delta", "product_gamma",
    "product_position_delta", "product_position_gamma", "pre_hedge_contracts",
    "post_hedge_contracts", "futures_multiplier", "pre_hedge_futures_delta",
    "post_hedge_futures_delta", "pre_hedge_delta", "post_hedge_delta",
    "pre_hedge_gamma", "post_hedge_gamma", "pre_hedge_delta_cash_1pct",
    "post_hedge_delta_cash_1pct", "pre_hedge_gamma_cash_1pct",
    "post_hedge_gamma_cash_1pct", "delta_cash_1pct", "gamma_cash_1pct",
    "vega", "theta", "rho", "dividend_sensitivity", "basis_sensitivity",
)

# Book-level greeks omit the per-product columns (the book nets positions).
BOOK_GREEK_COLUMNS: tuple[str, ...] = (
    "date", "delta", "gamma", "product_position_delta", "product_position_gamma",
    "pre_hedge_contracts", "post_hedge_contracts", "futures_multiplier",
    "pre_hedge_futures_delta", "post_hedge_futures_delta", "pre_hedge_delta",
    "post_hedge_delta", "pre_hedge_gamma", "post_hedge_gamma",
    "pre_hedge_delta_cash_1pct", "post_hedge_delta_cash_1pct",
    "pre_hedge_gamma_cash_1pct", "post_hedge_gamma_cash_1pct",
    "delta_cash_1pct", "gamma_cash_1pct",
)

TRADE_COLUMNS: tuple[str, ...] = (
    "date", "trade_type", "instrument_type", "contract", "quantity", "price",
    "multiplier", "notional", "transaction_cost", "reason",
)

REBALANCE_COLUMNS: tuple[str, ...] = (
    "date", "active_contract", "current_contracts", "target_contracts",
    "trade_contracts", "should_rebalance", "threshold_status",
    "no_trade_reason", "reason",
)

ACTION_COLUMNS: tuple[str, ...] = (
    "date", "action_type", "observation_index", "spot", "barrier", "cashflow",
    "alive_before", "knocked_in_before", "knocked_out_before", "matured_before",
    "alive_after", "knocked_in_after", "knocked_out_after", "matured_after",
)

DAILY_EVENT_COLUMNS: tuple[str, ...] = (
    "date", "next_ko_date", "next_ko_probability",
    "total_remaining_ko_probability", "ki_probability_to_maturity",
    "survival_probability", "expected_discounted_ko_cashflow",
    "expected_discounted_maturity_cashflow", "pv", "event_stats_engine",
)

EVENT_PROB_COLUMNS: tuple[str, ...] = (
    "date", "event_date", "event_type", "event_probability",
    "conditional_probability", "survival_probability",
    "expected_discounted_cashflow", "event_stats_engine",
)

SURFACE_COLUMNS: tuple[str, ...] = (
    "date", "surface_type", "spot_node", "q_node", "price", "delta", "gamma",
    "delta_cash_1pct", "gamma_cash_1pct",
)

CALIBRATION_RECORD_KEYS: tuple[str, ...] = (
    "date", "variant", "surface_date", "surface_sha", "cache_hit",
)


class StateRow(TypedDict):
    date: Any
    portfolio_value: float
    product_mtm: float
    hedge_mtm: float
    cash: float
    cashflows: float
    transaction_costs: float
    product_pnl: float
    hedge_pnl: float
    total_pnl: float
    spot: float
    volatility: float
    rate: float
    basis_yield: float
    implied_q: float
    pricing_q: float
    active_contract: str
    futures_price: float
    futures_ttm: float
    futures_multiplier: float
    futures_contracts: float
    alive: bool
    knocked_in: bool
    knocked_out: bool
    matured: bool


class GreekRow(TypedDict):
    date: Any
    price: float
    delta: float
    gamma: float
    product_delta: float
    product_gamma: float
    product_position_delta: float
    product_position_gamma: float
    pre_hedge_contracts: float
    post_hedge_contracts: float
    futures_multiplier: float
    pre_hedge_futures_delta: float
    post_hedge_futures_delta: float
    pre_hedge_delta: float
    post_hedge_delta: float
    pre_hedge_gamma: float
    post_hedge_gamma: float
    pre_hedge_delta_cash_1pct: float
    post_hedge_delta_cash_1pct: float
    pre_hedge_gamma_cash_1pct: float
    post_hedge_gamma_cash_1pct: float
    delta_cash_1pct: float
    gamma_cash_1pct: float
    vega: float
    theta: float
    rho: float
    dividend_sensitivity: float
    basis_sensitivity: float


class TradeRow(TypedDict):
    date: Any
    trade_type: str
    instrument_type: str
    contract: str
    quantity: float
    price: float
    multiplier: float
    notional: float
    transaction_cost: float
    reason: str


class RebalanceRow(TypedDict):
    date: Any
    active_contract: str
    current_contracts: float
    target_contracts: float
    trade_contracts: float
    should_rebalance: bool
    threshold_status: str
    no_trade_reason: Optional[str]
    reason: str


class ActionRow(TypedDict):
    date: Any
    action_type: str
    observation_index: Optional[int]
    spot: float
    barrier: Optional[float]
    cashflow: float
    alive_before: bool
    knocked_in_before: bool
    knocked_out_before: bool
    matured_before: bool
    alive_after: bool
    knocked_in_after: bool
    knocked_out_after: bool
    matured_after: bool


class DailyEventRow(TypedDict):
    date: Any
    next_ko_date: Any
    next_ko_probability: float
    total_remaining_ko_probability: float
    ki_probability_to_maturity: float
    survival_probability: float
    expected_discounted_ko_cashflow: float
    expected_discounted_maturity_cashflow: float
    pv: float
    event_stats_engine: str


class EventProbRow(TypedDict):
    date: Any
    event_date: Any
    event_type: str
    event_probability: float
    conditional_probability: float
    survival_probability: float
    expected_discounted_cashflow: float
    event_stats_engine: str


class SurfaceRow(TypedDict):
    date: Any
    surface_type: str
    spot_node: float
    q_node: float
    price: float
    delta: float
    gamma: float
    delta_cash_1pct: float
    gamma_cash_1pct: float


class CalibrationRecord(TypedDict, total=False):
    date: str
    variant: str
    surface_date: str
    surface_sha: str
    cache_hit: bool
