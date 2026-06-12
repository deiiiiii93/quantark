"""
Barrier-trigger (zoned) delta hedging strategy.

Desks hedge products with knock-in/knock-out features more aggressively as
spot approaches the dangerous level, and switch regime once the barrier
event happens:

    Spot level
    ^
    Safe zone        light hedge      (loose threshold)
    --------------
    Near barrier     hedge faster     (tight threshold)
    -------------- KI barrier
    Knocked in       new regime       (post-KI threshold)

The knock-in state is path-dependent and therefore *latched*: once spot
crosses the barrier, the strategy stays in the knocked-in regime even if
spot later recovers. State updates happen in on_step (called by the engine
every timestep before the hedge decision).
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from quantark.backtest.strategy.base_strategy import (
    AssetClass,
    BaseStrategy,
    HedgingTarget,
)
from quantark.util.exceptions import ValidationError

ZONE_FAR = "far"
ZONE_NEAR = "near_barrier"
ZONE_KNOCKED_IN = "knocked_in"


class BarrierTriggerHedgeStrategy(BaseStrategy):
    """
    Delta hedging with barrier-proximity zones and a knock-in regime.

    The delta threshold that triggers a hedge depends on the current zone:
    far from the barrier the book is allowed to drift (light hedging),
    near the barrier the threshold tightens (hedge faster), and after a
    knock-in event a third threshold applies (new regime).

    Attributes:
        barrier_level: Barrier level (e.g. knock-in put strike level)
        barrier_direction: 'down' (knock-in when spot falls to the barrier,
            the snowball case) or 'up'
        proximity_band: Relative distance |spot/barrier - 1| defining the
            near-barrier zone
        far_delta_threshold: Trigger threshold in the safe zone
        near_delta_threshold: Trigger threshold near the barrier
        post_ki_delta_threshold: Trigger threshold after knock-in
        target_delta: Target delta after hedging
    """

    VALID_DIRECTIONS = ["down", "up"]
    VALID_INSTRUMENTS = ["spot", "futures"]

    def __init__(
        self,
        barrier_level: float,
        name: str = "BarrierTrigger",
        barrier_direction: str = "down",
        proximity_band: float = 0.10,
        far_delta_threshold: float = 100.0,
        near_delta_threshold: float = 20.0,
        post_ki_delta_threshold: float = 50.0,
        target_delta: float = 0.0,
        hedge_instrument: str = "spot",
        min_time_between_hedges: Optional[timedelta] = None,
    ):
        """
        Initialize barrier-trigger hedging strategy.

        Args:
            barrier_level: Barrier level (> 0)
            name: Strategy name
            barrier_direction: 'down' or 'up'
            proximity_band: Relative width of the near-barrier zone (> 0)
            far_delta_threshold: Delta trigger in the safe zone
            near_delta_threshold: Delta trigger near the barrier
            post_ki_delta_threshold: Delta trigger after knock-in
            target_delta: Target delta after hedging
            hedge_instrument: 'spot' or 'futures'
            min_time_between_hedges: Minimum time between hedges

        Raises:
            ValidationError: If parameters are invalid
        """
        if barrier_level <= 0:
            raise ValidationError(
                f"barrier_level must be positive, got {barrier_level}"
            )
        if barrier_direction not in self.VALID_DIRECTIONS:
            raise ValidationError(
                f"Invalid barrier_direction '{barrier_direction}'. "
                f"Must be one of {self.VALID_DIRECTIONS}"
            )
        if proximity_band <= 0:
            raise ValidationError(
                f"proximity_band must be positive, got {proximity_band}"
            )
        for label, threshold in (
            ("far_delta_threshold", far_delta_threshold),
            ("near_delta_threshold", near_delta_threshold),
            ("post_ki_delta_threshold", post_ki_delta_threshold),
        ):
            if threshold < 0:
                raise ValidationError(
                    f"{label} must be non-negative, got {threshold}"
                )
        if hedge_instrument not in self.VALID_INSTRUMENTS:
            raise ValidationError(
                f"Invalid hedge_instrument '{hedge_instrument}'. "
                f"Must be one of {self.VALID_INSTRUMENTS}"
            )

        super().__init__(
            name=name,
            asset_class=AssetClass.EQUITY,
            hedging_target=HedgingTarget.DELTA,
            hedge_instrument=hedge_instrument,
        )

        self.barrier_level = barrier_level
        self.barrier_direction = barrier_direction
        self.proximity_band = proximity_band
        self.far_delta_threshold = far_delta_threshold
        self.near_delta_threshold = near_delta_threshold
        self.post_ki_delta_threshold = post_ki_delta_threshold
        self.target_delta = target_delta
        self.min_time_between_hedges = min_time_between_hedges

        # Path-dependent state
        self._knocked_in = False
        self._knock_in_time: Optional[datetime] = None
        self._hedge_count_by_zone: Dict[str, int] = {
            ZONE_FAR: 0,
            ZONE_NEAR: 0,
            ZONE_KNOCKED_IN: 0,
        }

    @property
    def knocked_in(self) -> bool:
        """Whether the barrier has been touched (latched)."""
        return self._knocked_in

    @property
    def delta_threshold(self) -> float:
        """Current far-zone threshold (engine logging compatibility)."""
        return self.far_delta_threshold

    def _crossed(self, spot: float) -> bool:
        """Whether this spot level constitutes a barrier crossing."""
        if self.barrier_direction == "down":
            return spot <= self.barrier_level
        return spot >= self.barrier_level

    def get_zone(self, market_data: Dict[str, float]) -> str:
        """
        Current hedging zone.

        Args:
            market_data: Current market data (uses 'spot')

        Returns:
            One of 'far', 'near_barrier', 'knocked_in'
        """
        if self._knocked_in:
            return ZONE_KNOCKED_IN
        spot = market_data.get("spot")
        if spot is not None and (
            abs(spot / self.barrier_level - 1.0) <= self.proximity_band
        ):
            return ZONE_NEAR
        return ZONE_FAR

    def _zone_threshold(self, zone: str) -> float:
        """Delta trigger threshold for a zone."""
        return {
            ZONE_FAR: self.far_delta_threshold,
            ZONE_NEAR: self.near_delta_threshold,
            ZONE_KNOCKED_IN: self.post_ki_delta_threshold,
        }[zone]

    def on_step(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ):
        """Latch the knock-in state when spot crosses the barrier."""
        spot = market_data.get("spot")
        if spot is not None and not self._knocked_in and self._crossed(spot):
            self._knocked_in = True
            self._knock_in_time = current_time

    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> bool:
        """Hedge when delta breaches the current zone's threshold."""
        if self.min_time_between_hedges is not None:
            time_since_hedge = self.time_since_last_hedge(current_time)
            if (
                time_since_hedge is not None
                and time_since_hedge < self.min_time_between_hedges
            ):
                return False

        deviation = portfolio_greeks.get("delta", 0.0) - self.target_delta
        zone = self.get_zone(market_data)
        return abs(deviation) > self._zone_threshold(zone)

    def calculate_hedge_size(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> float:
        """Trade the full delta deviation back to target."""
        deviation = portfolio_greeks.get("delta", 0.0) - self.target_delta
        # Track which zone triggered the hedge
        zone = self.get_zone(market_data)
        self._last_zone = zone
        return -deviation

    def on_hedge_executed(
        self, current_time: datetime, hedge_size: float, hedge_price: float, **kwargs
    ):
        """Update strategy state and per-zone statistics."""
        super().on_hedge_executed(current_time, hedge_size, hedge_price, **kwargs)
        zone = getattr(self, "_last_zone", ZONE_FAR)
        self._hedge_count_by_zone[zone] += 1

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            "name": self.name,
            "barrier_level": self.barrier_level,
            "barrier_direction": self.barrier_direction,
            "proximity_band": self.proximity_band,
            "far_delta_threshold": self.far_delta_threshold,
            "near_delta_threshold": self.near_delta_threshold,
            "post_ki_delta_threshold": self.post_ki_delta_threshold,
            "target_delta": self.target_delta,
            "hedge_instrument": self.hedge_instrument,
            "min_time_between_hedges": (
                str(self.min_time_between_hedges)
                if self.min_time_between_hedges
                else None
            ),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            "knocked_in": self._knocked_in,
            "knock_in_time": self._knock_in_time,
            "hedge_count_by_zone": dict(self._hedge_count_by_zone),
            "last_hedge_time": self._last_hedge_time,
        }

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._knocked_in = False
        self._knock_in_time = None
        self._hedge_count_by_zone = {
            ZONE_FAR: 0,
            ZONE_NEAR: 0,
            ZONE_KNOCKED_IN: 0,
        }

    def __repr__(self) -> str:
        return (
            f"BarrierTriggerHedgeStrategy("
            f"barrier={self.barrier_level}, "
            f"direction={self.barrier_direction}, "
            f"knocked_in={self._knocked_in})"
        )
