"""
Credit spread-neutral (CS01-neutral) hedging strategy.

Monitors a credit book's net CS01 (per reference entity) and neutralises it
with an offsetting credit hedge (e.g. an index CDS) whenever it breaches a
threshold. Mirrors the structure of FXDeltaNeutralStrategy but targets CS01.
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


class CreditSpreadNeutralStrategy(BaseStrategy):
    """Hedge net CS01 to a target (default 0) using an offsetting credit position."""

    VALID_FREQUENCIES = ["daily", "hourly", "on_threshold", "continuous"]
    VALID_INSTRUMENTS = ["cds", "index"]

    def __init__(
        self,
        name: str = "CreditSpreadNeutral",
        cs01_threshold: float = 5_000.0,
        rebalance_frequency: str = "on_threshold",
        hedge_instrument: str = "cds",
        hedge_ratio: float = 1.0,
        target_cs01: float = 0.0,
        min_time_between_hedges: Optional[timedelta] = None,
    ):
        super().__init__(
            name=name,
            asset_class=AssetClass.GENERIC,
            hedging_target=HedgingTarget.DELTA,
            hedge_instrument=hedge_instrument,
        )
        if cs01_threshold < 0:
            raise ValidationError(
                f"cs01_threshold must be non-negative, got {cs01_threshold}"
            )
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

        self.cs01_threshold = cs01_threshold
        self.rebalance_frequency = rebalance_frequency
        self.hedge_ratio = hedge_ratio
        self.target_cs01 = target_cs01
        self.min_time_between_hedges = min_time_between_hedges
        self._hedge_count = 0

    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> bool:
        net_cs01 = portfolio_greeks.get("cs01", 0.0)
        breach = abs(net_cs01 - self.target_cs01) > self.cs01_threshold
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
        """CS01 amount of hedge to add to bring net CS01 to target."""
        net_cs01 = portfolio_greeks.get("cs01", 0.0)
        return -(net_cs01 - self.target_cs01) * self.hedge_ratio

    def on_hedge_executed(self, current_time, hedge_size, hedge_price, **kwargs):
        super().on_hedge_executed(current_time, hedge_size, hedge_price, **kwargs)
        self._hedge_count += 1

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hedging_target": self.hedging_target.value,
            "cs01_threshold": self.cs01_threshold,
            "rebalance_frequency": self.rebalance_frequency,
            "hedge_instrument": self.hedge_instrument,
            "hedge_ratio": self.hedge_ratio,
            "target_cs01": self.target_cs01,
        }

    def reset(self):
        super().reset()
        self._hedge_count = 0

    def __repr__(self) -> str:
        return (
            f"CreditSpreadNeutralStrategy(threshold={self.cs01_threshold:,.0f}, "
            f"freq={self.rebalance_frequency})"
        )
