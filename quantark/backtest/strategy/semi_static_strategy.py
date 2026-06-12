"""
Semi-static (event-driven) hedging strategy.

Sits between static and dynamic hedging: the hedge portfolio is left alone
between key events and rebalanced only at:

- the trade date (initial hedge)
- scheduled dates (observation dates, coupon dates, expiry)
- barrier proximity (risk changes sharply near knock-in/out levels)

    Time ----------------------------------------------->
    Trade     Obs 1     Obs 2     Obs 3     Expiry
      |         |         |         |         |
     hedge    adjust    adjust    adjust    final

This is common for snowballs/autocallables, whose Greeks jump around
observation dates and barriers while staying comparatively stable in
between. Sizing is inherited from MultiGreekHedgeStrategy, so the event
gate composes with any target set: the default is a plain delta hedge
(spot only, a 1x1 solve), but delta+gamma+vega at each observation date is
just a different target/instrument list.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from quantark.backtest.strategy.base_strategy import passes_frequency_gate
from quantark.backtest.strategy.hedge_instruments import (
    BaseHedgeInstrument,
    SpotHedgeInstrument,
)
from quantark.backtest.strategy.hedge_optimizer import HedgeTarget
from quantark.backtest.strategy.multi_greek_strategy import MultiGreekHedgeStrategy
from quantark.util.exceptions import ValidationError


class SemiStaticHedgeStrategy(MultiGreekHedgeStrategy):
    """
    Rebalances only at key events instead of continuously.

    Events, in order of evaluation:
    1. Trade date: the first backtest step (if hedge_at_start)
    2. Barrier proximity: every step while |spot/barrier - 1| <= band
    3. Scheduled dates: once per scheduled calendar date

    Outside events the book is left alone regardless of Greek drift.

    Attributes:
        rebalance_dates: Scheduled rebalance dates (observation, coupon,
            expiry dates), matched by calendar date
        hedge_at_start: Whether to hedge on the first backtest step
        barrier_level: Optional barrier whose proximity forces rebalancing
        barrier_proximity_band: Relative distance to the barrier inside
            which every step is an event (0.05 = within 5%)
    """

    def __init__(
        self,
        name: str = "SemiStatic",
        targets: Optional[List[HedgeTarget]] = None,
        hedge_instruments: Optional[List[BaseHedgeInstrument]] = None,
        rebalance_dates: Optional[List[datetime]] = None,
        hedge_at_start: bool = True,
        barrier_level: Optional[float] = None,
        barrier_proximity_band: float = 0.05,
        min_time_between_hedges: Optional[timedelta] = None,
        instrument_costs: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize semi-static hedging strategy.

        Args:
            name: Strategy name
            targets: Greeks to control at each event (default: delta to 0
                with zero threshold, i.e. fully rebalance at every event)
            hedge_instruments: Instruments to trade (default: spot only)
            rebalance_dates: Scheduled event dates (observation/coupon/expiry)
            hedge_at_start: Hedge on the first backtest step (trade date)
            barrier_level: Optional barrier level forcing nearby rebalancing
            barrier_proximity_band: Relative proximity band around the barrier
            min_time_between_hedges: Minimum time between hedges
            instrument_costs: Optional cost weight per instrument name

        Raises:
            ValidationError: If parameters are invalid
        """
        if barrier_level is not None and barrier_level <= 0:
            raise ValidationError(
                f"barrier_level must be positive, got {barrier_level}"
            )
        if barrier_proximity_band <= 0:
            raise ValidationError(
                f"barrier_proximity_band must be positive, "
                f"got {barrier_proximity_band}"
            )

        if targets is None:
            targets = [HedgeTarget("delta")]
        if hedge_instruments is None:
            hedge_instruments = [SpotHedgeInstrument()]

        super().__init__(
            name=name,
            targets=targets,
            hedge_instruments=hedge_instruments,
            rebalance_frequency="continuous",  # gating is event-driven below
            min_time_between_hedges=min_time_between_hedges,
            instrument_costs=instrument_costs,
        )

        self.rebalance_dates = list(rebalance_dates or [])
        self.hedge_at_start = hedge_at_start
        self.barrier_level = barrier_level
        self.barrier_proximity_band = barrier_proximity_band

        self._scheduled_dates = {d.date() for d in self.rebalance_dates}
        self._first_step_time: Optional[datetime] = None

    def on_step(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ):
        """Record the first backtest step (the trade date)."""
        if self._first_step_time is None:
            self._first_step_time = current_time

    def is_near_barrier(self, market_data: Dict[str, float]) -> bool:
        """Whether spot is inside the proximity band around the barrier."""
        if self.barrier_level is None:
            return False
        spot = market_data.get("spot")
        if spot is None:
            return False
        return abs(spot / self.barrier_level - 1.0) <= self.barrier_proximity_band

    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> bool:
        """
        Hedge only at events, and only if some target actually deviates.
        """
        if self.min_time_between_hedges is not None:
            time_since_hedge = self.time_since_last_hedge(current_time)
            if (
                time_since_hedge is not None
                and time_since_hedge < self.min_time_between_hedges
            ):
                return False

        deviations = self.get_target_deviations(portfolio_greeks)
        breach = any(abs(deviations[t.greek]) > t.threshold for t in self.targets)
        if not breach:
            return False

        # Trade date: hedge on the very first step
        if (
            self.hedge_at_start
            and self._first_step_time is not None
            and current_time == self._first_step_time
        ):
            return True

        # Barrier proximity: every step is an event while near the barrier
        if self.is_near_barrier(market_data):
            return True

        # Scheduled events: once per scheduled calendar date
        if current_time.date() in self._scheduled_dates:
            return passes_frequency_gate(
                frequency="daily",
                breach=True,
                current_time=current_time,
                last_rebalance_time=self._last_rebalance_date,
            )

        return False

    def get_parameters(self) -> Dict[str, object]:
        """Get strategy parameters."""
        params = super().get_parameters()
        params.update(
            {
                "rebalance_dates": [d.isoformat() for d in self.rebalance_dates],
                "hedge_at_start": self.hedge_at_start,
                "barrier_level": self.barrier_level,
                "barrier_proximity_band": self.barrier_proximity_band,
            }
        )
        return params

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._first_step_time = None
