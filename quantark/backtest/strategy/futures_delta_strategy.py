"""
Delta-neutral futures hedge sizing for product-replay backtests.

Moved from ``quantark.backtest.otc.state`` into the shared strategy
hierarchy; sizing math is unchanged. The replay engines drive the native
``target_contracts`` / ``should_rebalance`` API; the ``BaseStrategy``
protocol methods are thin adapters over the same math so the strategy is
interchangeable with the rest of the hierarchy.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from quantark.util.exceptions import ValidationError

from .base_strategy import AssetClass, BaseStrategy, HedgingTarget


class AutocallableDeltaHedgeStrategy(BaseStrategy):
    """
    Delta-neutral futures hedge sizing.

    Product delta is assumed to be currency value per one index point. A futures
    contract contributes approximately ``multiplier`` per one index point.
    """

    def __init__(
        self,
        delta_threshold: float = 0.0,
        hedge_ratio: float = 1.0,
        target_delta: float = 0.0,
        round_contracts: bool = True,
    ) -> None:
        super().__init__(
            name="AutocallableDeltaHedge",
            asset_class=AssetClass.EQUITY,
            hedging_target=HedgingTarget.DELTA,
            hedge_instrument="futures",
        )
        if delta_threshold < 0:
            raise ValidationError("delta_threshold must be non-negative")
        if not 0 <= hedge_ratio <= 1:
            raise ValidationError("hedge_ratio must be between 0 and 1")
        self.delta_threshold = delta_threshold
        self.hedge_ratio = hedge_ratio
        self.target_delta = target_delta
        self.round_contracts = round_contracts

    # ------------------------------------------------------------------
    # Native replay API (unchanged semantics)
    # ------------------------------------------------------------------

    def target_contracts(
        self,
        *,
        product_delta: float,
        product_quantity: float,
        futures_multiplier: float,
    ) -> float:
        if futures_multiplier <= 0:
            raise ValidationError("futures_multiplier must be positive")
        net_delta = float(product_delta) * float(product_quantity)
        target = -((net_delta - self.target_delta) / float(futures_multiplier))
        target *= self.hedge_ratio
        if self.round_contracts:
            return float(round(target))
        return float(target)

    def should_rebalance(self, current_contracts: float, target_contracts: float) -> bool:
        return abs(float(target_contracts) - float(current_contracts)) > self.delta_threshold

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "delta_threshold": self.delta_threshold,
            "hedge_ratio": self.hedge_ratio,
            "target_delta": self.target_delta,
            "round_contracts": self.round_contracts,
        }

    # ------------------------------------------------------------------
    # BaseStrategy protocol adapters
    # ------------------------------------------------------------------

    def _multiplier_from(self, market_data: Optional[Dict[str, Any]]) -> float:
        multiplier = (market_data or {}).get("futures_multiplier")
        if multiplier is None:
            raise ValidationError(
                "market_data['futures_multiplier'] is required to size a "
                "futures delta hedge"
            )
        return float(multiplier)

    def calculate_hedge_size(
        self, current_time, portfolio_greeks, market_data, **kwargs
    ) -> float:
        """Target contract count for the net portfolio delta."""
        return self.target_contracts(
            product_delta=float(portfolio_greeks.get("delta", 0.0)),
            product_quantity=1.0,
            futures_multiplier=self._multiplier_from(market_data),
        )

    def should_hedge(
        self, current_time, portfolio_greeks, market_data, **kwargs
    ) -> bool:
        """True when the target differs from the current holding by more than
        the contract band (``current_contracts`` defaults to flat)."""
        current = float(kwargs.get("current_contracts", 0.0))
        target = self.calculate_hedge_size(
            current_time, portfolio_greeks, market_data
        )
        return self.should_rebalance(current, target)


class ProportionalFuturesDeltaHedgeStrategy(AutocallableDeltaHedgeStrategy):
    """Single-contract delta hedge with the ``S/F`` scaling correction.

    The legacy strategy above holds ``h_j = -D/m_j`` and therefore keeps a
    residual spot delta of ``D(1 - F_j/S)``: one futures contract moves by
    ``m_j dF_j``, and ``dF_j = (F_j/S) dS`` at a frozen carry curve, so
    sizing off the multiplier alone under- or over-hedges by the basis.

    This class holds ``h_j = -D S / (m_j F_j)`` instead, which is spot
    neutral and leaves ``R_i + 1_(i=j) D S T_j`` of carry risk.  Separating it
    from the bucket policies is the point: it isolates the effect of
    correcting the scaling from the effect of holding a calendar spread.

    It is a DISTINCT policy, not a change to the legacy sizing.  The parent's
    ``target_contracts`` keeps its own signature and output, and both scalar
    protocol adapters are overridden here so an inherited caller cannot reach
    the new method without supplying the two prices it needs.
    """

    def __init__(
        self,
        delta_threshold: float = 0.0,
        hedge_ratio: float = 1.0,
        target_delta: float = 0.0,
        round_contracts: bool = True,
    ) -> None:
        super().__init__(
            delta_threshold=delta_threshold,
            hedge_ratio=hedge_ratio,
            target_delta=target_delta,
            round_contracts=round_contracts,
        )
        self.name = "ProportionalFuturesDeltaHedge"

    def target_contracts(
        self,
        *,
        product_delta: float,
        product_quantity: float,
        futures_multiplier: float,
        spot: float,
        futures_price: float,
    ) -> float:
        values = (futures_multiplier, spot, futures_price)
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
            raise ValidationError(
                "multiplier, spot and futures price must be positive"
            )
        target = (
            -float(product_delta)
            * float(product_quantity)
            * float(spot)
            / (float(futures_multiplier) * float(futures_price))
        )
        target *= self.hedge_ratio
        return float(round(target)) if self.round_contracts else float(target)

    def _prices_from(self, market_data: Optional[Dict[str, Any]]) -> tuple:
        data = market_data or {}
        spot = data.get("spot")
        futures_price = data.get("futures_price")
        if spot is None or futures_price is None:
            raise ValidationError(
                "market_data['spot'] and market_data['futures_price'] are "
                "required to size a proportional futures delta hedge"
            )
        return float(spot), float(futures_price)

    def calculate_hedge_size(
        self, current_time, portfolio_greeks, market_data, **kwargs
    ) -> float:
        """Target contracts for the net portfolio delta, at ``S/F`` scaling."""
        spot, futures_price = self._prices_from(market_data)
        return self.target_contracts(
            product_delta=float(portfolio_greeks.get("delta", 0.0)),
            product_quantity=1.0,
            futures_multiplier=self._multiplier_from(market_data),
            spot=spot,
            futures_price=futures_price,
        )
