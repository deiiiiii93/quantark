"""
FX delta-neutral hedging strategy.

Monitors an FX book's spot delta (per currency pair) and neutralises it with an
FX spot/forward hedge whenever it breaches a threshold. Mirrors the structure of
DV01NeutralStrategy but targets FX delta.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from quantark.backtest.strategy.base_strategy import (
    AssetClass,
    BaseStrategy,
    HedgingTarget,
    passes_frequency_gate,
)
from quantark.util.exceptions import ValidationError


class FXDeltaNeutralStrategy(BaseStrategy):
    """Hedge FX spot delta to a target (default 0) using an FX spot position."""

    VALID_FREQUENCIES = ["daily", "hourly", "on_threshold", "continuous"]
    VALID_INSTRUMENTS = ["spot", "forward"]

    def __init__(
        self,
        name: str = "FXDeltaNeutral",
        delta_threshold: float = 50_000.0,
        rebalance_frequency: str = "on_threshold",
        hedge_instrument: str = "spot",
        hedge_ratio: float = 1.0,
        target_delta: float = 0.0,
        min_time_between_hedges: Optional[timedelta] = None,
    ):
        super().__init__(
            name=name,
            asset_class=AssetClass.GENERIC,
            hedging_target=HedgingTarget.DELTA,
            hedge_instrument=hedge_instrument,
        )
        if delta_threshold < 0:
            raise ValidationError(f"delta_threshold must be non-negative, got {delta_threshold}")
        if rebalance_frequency not in self.VALID_FREQUENCIES:
            raise ValidationError(
                f"Invalid rebalance_frequency '{rebalance_frequency}'. "
                f"Must be one of {self.VALID_FREQUENCIES}")
        if hedge_instrument not in self.VALID_INSTRUMENTS:
            raise ValidationError(
                f"Invalid hedge_instrument '{hedge_instrument}'. "
                f"Must be one of {self.VALID_INSTRUMENTS}")
        if not 0 <= hedge_ratio <= 1:
            raise ValidationError(f"hedge_ratio must be in [0, 1], got {hedge_ratio}")

        self.delta_threshold = delta_threshold
        self.rebalance_frequency = rebalance_frequency
        self.hedge_ratio = hedge_ratio
        self.target_delta = target_delta
        self.min_time_between_hedges = min_time_between_hedges
        self._hedge_count = 0

    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> bool:
        net_delta = portfolio_greeks.get("delta", 0.0)
        breach = abs(net_delta - self.target_delta) > self.delta_threshold
        if self.min_time_between_hedges is not None:
            elapsed = self.time_since_last_hedge(current_time)
            if elapsed is not None and elapsed < self.min_time_between_hedges:
                return False
        return passes_frequency_gate(
            self.rebalance_frequency, breach, current_time, self._last_hedge_time
        )

    def calculate_hedge_size(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> float:
        """Base-currency notional to trade to bring net delta to target."""
        net_delta = portfolio_greeks.get("delta", 0.0)
        return -(net_delta - self.target_delta) * self.hedge_ratio

    def on_hedge_executed(self, current_time, hedge_size, hedge_price, **kwargs):
        super().on_hedge_executed(current_time, hedge_size, hedge_price, **kwargs)
        self._hedge_count += 1

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hedging_target": self.hedging_target.value,
            "delta_threshold": self.delta_threshold,
            "rebalance_frequency": self.rebalance_frequency,
            "hedge_instrument": self.hedge_instrument,
            "hedge_ratio": self.hedge_ratio,
            "target_delta": self.target_delta,
        }

    def reset(self):
        super().reset()
        self._hedge_count = 0

    def __repr__(self) -> str:
        return (
            f"FXDeltaNeutralStrategy(threshold={self.delta_threshold:,.0f}, "
            f"freq={self.rebalance_frequency})"
        )
