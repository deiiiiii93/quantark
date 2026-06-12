"""
Whalley-Wilmott utility-based delta hedging band strategy.

Whalley & Wilmott (1997) derived the asymptotically optimal hedging policy
under proportional transaction costs and exponential utility: do not trade
while the portfolio delta stays inside a no-transaction band around the
target, and when the band is breached, trade back to the nearest band
boundary (not the center). The band half-width is

    H = ( (3/2) * k * S * Gamma^2 * e^{-r * tau} / lambda )^(1/3)

where k is the proportional transaction cost rate, S the spot, Gamma the
portfolio gamma, lambda the risk aversion and tau the remaining option
maturity. If horizon (tau) is not supplied, the undiscounted leading-order
band H = ((3/2) * k * S * Gamma^2 / lambda)^(1/3) is used, as commonly
quoted in the literature (e.g. Zakamouline's surveys).

The band scales with Gamma^(2/3): large books rebalance relatively less
often, and the band collapses to pure threshold hedging as Gamma -> 0.
"""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from quantark.backtest.strategy.base_strategy import (
    AssetClass,
    BaseStrategy,
    HedgingTarget,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_exp


class WhalleyWilmottStrategy(BaseStrategy):
    """
    Transaction-cost-aware delta hedging with a utility-based no-trade band.

    Attributes:
        risk_aversion: Exponential utility risk aversion (lambda > 0);
            higher values mean tighter bands and more frequent hedging
        cost_rate: Proportional transaction cost rate k (e.g. 0.001 = 10 bps)
        target_delta: Center of the no-trade band
        horizon: Optional remaining option maturity tau in years used for
            the e^{-r * tau} discount term; None uses the undiscounted band
        rebalance_to: 'boundary' (Whalley-Wilmott optimal) trades back to
            the nearest band edge; 'target' trades all the way to target_delta
        min_time_between_hedges: Minimum time between hedges
    """

    VALID_REBALANCE_TO = ["boundary", "target"]
    VALID_INSTRUMENTS = ["spot", "futures"]

    def __init__(
        self,
        name: str = "WhalleyWilmott",
        risk_aversion: float = 1.0,
        cost_rate: float = 0.001,
        target_delta: float = 0.0,
        horizon: Optional[float] = None,
        rebalance_to: str = "boundary",
        hedge_instrument: str = "spot",
        min_time_between_hedges: Optional[timedelta] = None,
    ):
        """
        Initialize Whalley-Wilmott strategy.

        Args:
            name: Strategy name
            risk_aversion: Risk aversion lambda (> 0)
            cost_rate: Proportional transaction cost rate k (>= 0)
            target_delta: Center of the no-trade band
            horizon: Optional remaining maturity tau (years) for discounting
            rebalance_to: 'boundary' or 'target'
            hedge_instrument: 'spot' or 'futures'
            min_time_between_hedges: Minimum time between hedges

        Raises:
            ValidationError: If parameters are invalid
        """
        if risk_aversion <= 0:
            raise ValidationError(
                f"risk_aversion must be positive, got {risk_aversion}"
            )
        if cost_rate < 0:
            raise ValidationError(
                f"cost_rate must be non-negative, got {cost_rate}"
            )
        if horizon is not None and horizon < 0:
            raise ValidationError(f"horizon must be non-negative, got {horizon}")
        if rebalance_to not in self.VALID_REBALANCE_TO:
            raise ValidationError(
                f"Invalid rebalance_to '{rebalance_to}'. "
                f"Must be one of {self.VALID_REBALANCE_TO}"
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

        self.risk_aversion = risk_aversion
        self.cost_rate = cost_rate
        self.target_delta = target_delta
        self.horizon = horizon
        self.rebalance_to = rebalance_to
        self.min_time_between_hedges = min_time_between_hedges

        # Internal state
        self._hedge_count = 0

    def get_band_half_width(
        self,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
    ) -> float:
        """
        Half-width H of the no-transaction band.

        Args:
            portfolio_greeks: Current portfolio Greeks (uses gamma)
            market_data: Current market data (uses spot and, if a horizon
                is configured, rate)

        Returns:
            Band half-width in delta units
        """
        spot = market_data.get("spot", 0.0)
        gamma = portfolio_greeks.get("gamma", 0.0)

        discount = 1.0
        if self.horizon is not None:
            rate = market_data.get("rate", 0.0)
            discount = float(safe_exp(-rate * self.horizon))

        return float(
            1.5 * self.cost_rate * spot * gamma**2 * discount / self.risk_aversion
        ) ** (1.0 / 3.0)

    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> bool:
        """Hedge when delta leaves the no-transaction band."""
        if self.min_time_between_hedges is not None:
            time_since_hedge = self.time_since_last_hedge(current_time)
            if (
                time_since_hedge is not None
                and time_since_hedge < self.min_time_between_hedges
            ):
                return False

        deviation = portfolio_greeks.get("delta", 0.0) - self.target_delta
        band = self.get_band_half_width(portfolio_greeks, market_data)
        return abs(deviation) > band

    def calculate_hedge_size(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> float:
        """
        Trade back to the nearest band boundary (or to the target).

        For rebalance_to='boundary' and deviation d with band H:
            hedge = -(d - sign(d) * H)
        which leaves the portfolio exactly on the breached band edge.
        """
        deviation = portfolio_greeks.get("delta", 0.0) - self.target_delta

        if self.rebalance_to == "target":
            return -deviation

        band = self.get_band_half_width(portfolio_greeks, market_data)
        if abs(deviation) <= band:
            return 0.0
        return -(deviation - math.copysign(band, deviation))

    def on_hedge_executed(
        self, current_time: datetime, hedge_size: float, hedge_price: float, **kwargs
    ):
        """Update strategy state after hedge execution."""
        super().on_hedge_executed(current_time, hedge_size, hedge_price, **kwargs)
        self._hedge_count += 1

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            "name": self.name,
            "risk_aversion": self.risk_aversion,
            "cost_rate": self.cost_rate,
            "target_delta": self.target_delta,
            "horizon": self.horizon,
            "rebalance_to": self.rebalance_to,
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
            "hedge_count": self._hedge_count,
            "last_hedge_time": self._last_hedge_time,
        }

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._hedge_count = 0

    def __repr__(self) -> str:
        return (
            f"WhalleyWilmottStrategy("
            f"risk_aversion={self.risk_aversion}, "
            f"cost_rate={self.cost_rate}, "
            f"rebalance_to={self.rebalance_to})"
        )
