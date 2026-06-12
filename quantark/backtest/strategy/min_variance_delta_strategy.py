"""
Minimum-variance delta hedging strategy.

In equity markets, spot and implied volatility are correlated (spot down,
vol up for index skew), so the Black-Scholes delta is not the hedge ratio
that minimizes the variance of hedged P&L. The minimum-variance delta is

    delta_MV = delta_BS + vega_BS * d(sigma_imp)/dS

(Hull & White, "Optimal Delta Hedging for Options", 2017). The vol-spot
slope d(sigma)/dS must be supplied by the user, either as a constant or a
callable evaluated on each step's market data (e.g. fitted from the skew
or from a regression of implied vol changes on spot returns).

Unit conventions: the slope is in absolute volatility per unit of spot
(e.g. -0.001 means implied vol falls 10 bps when spot rises by 1.0).
QuantArk portfolio vega is reported per 1 percentage point of volatility,
so internally the adjustment uses vega * 100 * slope.
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Union

from quantark.backtest.strategy.base_strategy import (
    VALID_REBALANCE_FREQUENCIES,
    AssetClass,
    BaseStrategy,
    HedgingTarget,
    passes_frequency_gate,
)
from quantark.util.exceptions import ValidationError

VolSpotSlope = Union[float, Callable[[Dict[str, float]], float]]


class MinimumVarianceDeltaStrategy(BaseStrategy):
    """
    Delta hedging using the skew-adjusted minimum-variance delta.

    Behaves like DeltaNeutralStrategy, but thresholds and hedge sizes are
    computed on delta_MV = delta + vega_per_unit_vol * vol_spot_slope
    instead of the raw model delta.

    Attributes:
        delta_threshold: Absolute minimum-variance delta deviation that
            triggers a hedge
        vol_spot_slope: d(sigma_imp)/dS, constant or callable(market_data)
        target_delta: Target minimum-variance delta after hedging
        rebalance_frequency: Frequency gating mode
        min_time_between_hedges: Minimum time between hedges
    """

    VALID_INSTRUMENTS = ["spot", "futures"]

    def __init__(
        self,
        vol_spot_slope: VolSpotSlope,
        name: str = "MinimumVarianceDelta",
        delta_threshold: float = 100.0,
        target_delta: float = 0.0,
        rebalance_frequency: str = "daily",
        hedge_instrument: str = "spot",
        min_time_between_hedges: Optional[timedelta] = None,
    ):
        """
        Initialize minimum-variance delta strategy.

        Args:
            vol_spot_slope: d(sigma_imp)/dS in absolute vol per spot unit,
                either a constant or a callable taking market_data
            name: Strategy name
            delta_threshold: Absolute MV-delta deviation to trigger hedge
            target_delta: Target MV delta after hedging
            rebalance_frequency: 'daily', 'hourly', 'on_threshold', 'continuous'
            hedge_instrument: 'spot' or 'futures'
            min_time_between_hedges: Minimum time between hedges

        Raises:
            ValidationError: If parameters are invalid
        """
        if not callable(vol_spot_slope) and not isinstance(
            vol_spot_slope, (int, float)
        ):
            raise ValidationError(
                "vol_spot_slope must be a number or a callable taking "
                f"market_data, got {type(vol_spot_slope).__name__}"
            )
        if delta_threshold < 0:
            raise ValidationError(
                f"delta_threshold must be non-negative, got {delta_threshold}"
            )
        if rebalance_frequency not in VALID_REBALANCE_FREQUENCIES:
            raise ValidationError(
                f"Invalid rebalance_frequency '{rebalance_frequency}'. "
                f"Must be one of {VALID_REBALANCE_FREQUENCIES}"
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

        self.vol_spot_slope = vol_spot_slope
        self.delta_threshold = delta_threshold
        self.target_delta = target_delta
        self.rebalance_frequency = rebalance_frequency
        self.min_time_between_hedges = min_time_between_hedges

        # Internal state
        self._hedge_count = 0
        self._last_rebalance_date: Optional[datetime] = None

    def get_min_variance_delta(
        self,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
    ) -> float:
        """
        Minimum-variance delta of the portfolio.

        delta_MV = delta + vega_per_unit_vol * d(sigma)/dS, where reported
        vega (per 1% vol) is rescaled by 100 to per-unit-vol terms.

        Args:
            portfolio_greeks: Current portfolio Greeks
            market_data: Current market data (passed to a callable slope)

        Returns:
            Minimum-variance delta
        """
        delta = portfolio_greeks.get("delta", 0.0)
        vega_per_pct = portfolio_greeks.get("vega", 0.0)

        if callable(self.vol_spot_slope):
            slope = float(self.vol_spot_slope(market_data))
        else:
            slope = float(self.vol_spot_slope)

        return delta + vega_per_pct * 100.0 * slope

    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> bool:
        """Hedge when the minimum-variance delta breaches the threshold."""
        if self.min_time_between_hedges is not None:
            time_since_hedge = self.time_since_last_hedge(current_time)
            if (
                time_since_hedge is not None
                and time_since_hedge < self.min_time_between_hedges
            ):
                return False

        deviation = (
            self.get_min_variance_delta(portfolio_greeks, market_data)
            - self.target_delta
        )
        return passes_frequency_gate(
            frequency=self.rebalance_frequency,
            breach=abs(deviation) > self.delta_threshold,
            current_time=current_time,
            last_rebalance_time=self._last_rebalance_date,
        )

    def calculate_hedge_size(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> float:
        """Trade the minimum-variance delta deviation back to target."""
        deviation = (
            self.get_min_variance_delta(portfolio_greeks, market_data)
            - self.target_delta
        )
        return -deviation

    def on_hedge_executed(
        self, current_time: datetime, hedge_size: float, hedge_price: float, **kwargs
    ):
        """Update strategy state after hedge execution."""
        super().on_hedge_executed(current_time, hedge_size, hedge_price, **kwargs)
        self._hedge_count += 1
        self._last_rebalance_date = current_time

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            "name": self.name,
            "vol_spot_slope": (
                "callable" if callable(self.vol_spot_slope) else self.vol_spot_slope
            ),
            "delta_threshold": self.delta_threshold,
            "target_delta": self.target_delta,
            "rebalance_frequency": self.rebalance_frequency,
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
            "last_rebalance_date": self._last_rebalance_date,
        }

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._hedge_count = 0
        self._last_rebalance_date = None

    def __repr__(self) -> str:
        slope = "callable" if callable(self.vol_spot_slope) else self.vol_spot_slope
        return (
            f"MinimumVarianceDeltaStrategy("
            f"slope={slope}, threshold={self.delta_threshold})"
        )
