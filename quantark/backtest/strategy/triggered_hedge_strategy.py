"""
Trigger-armed (contingent) hedging strategy.

The hedge stays inactive until a *realized* market move from the start-date
reference fires a trigger — e.g. spot down 10% from inception, or implied
vol up 8 points. Once armed, the strategy behaves like an ordinary
multi-Greek threshold hedge: Greeks are neutralized back inside their
acceptable thresholds.

This is the contingent counterpart of ScenarioHedgeStrategy: there the
scenario is the *target* (pre-emptively flatten P&L under hypothetical
moves, paying hedge costs up front); here the realized move is the
*trigger* and Greeks are the target (pay hedging costs only once the book
is actually in danger).

Triggers are evaluated against reference values captured on the first
backtest step (the trade date). Multiple conditions inside one
HedgeTrigger must all hold jointly; the strategy arms when ANY of its
triggers fires.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from quantark.backtest.strategy.hedge_instruments import (
    BaseHedgeInstrument,
    SpotHedgeInstrument,
)
from quantark.backtest.strategy.hedge_optimizer import HedgeTarget
from quantark.backtest.strategy.multi_greek_strategy import MultiGreekHedgeStrategy
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class HedgeTrigger:
    """
    One realized-move condition that arms a TriggeredHedgeStrategy.

    All specified conditions must hold jointly for the trigger to fire
    (e.g. spot_drawdown=0.10 with vol_increase=0.05 means spot fell at
    least 10% AND vol rose at least 5 points).

    Attributes:
        name: Trigger identifier
        spot_drawdown: Fires when spot <= reference * (1 - x), x in (0, 1)
        spot_rally: Fires when spot >= reference * (1 + x), x > 0
        vol_increase: Fires when vol >= reference vol + x (absolute), x > 0
        vol_decrease: Fires when vol <= reference vol - x (absolute), x > 0
    """

    name: str
    spot_drawdown: Optional[float] = None
    spot_rally: Optional[float] = None
    vol_increase: Optional[float] = None
    vol_decrease: Optional[float] = None

    def __post_init__(self):
        if not self.name:
            raise ValidationError("HedgeTrigger name must be non-empty")
        conditions = (
            self.spot_drawdown,
            self.spot_rally,
            self.vol_increase,
            self.vol_decrease,
        )
        if all(c is None for c in conditions):
            raise ValidationError(
                f"HedgeTrigger '{self.name}' has no conditions; specify at "
                "least one of spot_drawdown/spot_rally/vol_increase/vol_decrease"
            )
        if self.spot_drawdown is not None and not 0 < self.spot_drawdown < 1:
            raise ValidationError(
                f"spot_drawdown must be in (0, 1), got {self.spot_drawdown}"
            )
        for label, value in (
            ("spot_rally", self.spot_rally),
            ("vol_increase", self.vol_increase),
            ("vol_decrease", self.vol_decrease),
        ):
            if value is not None and value <= 0:
                raise ValidationError(
                    f"{label} must be positive, got {value}"
                )

    def _require(
        self, key: str, data: Dict[str, float], which: str
    ) -> float:
        value = data.get(key)
        if value is None:
            raise ValidationError(
                f"HedgeTrigger '{self.name}' needs '{key}' in the {which} "
                "market data to evaluate its conditions"
            )
        return value

    def is_met(
        self, reference: Dict[str, float], market_data: Dict[str, float]
    ) -> bool:
        """
        Evaluate the trigger against current market data.

        Args:
            reference: Market data captured on the trade date
            market_data: Current market data

        Returns:
            True if all specified conditions hold

        Raises:
            ValidationError: If required market data is missing
        """
        if self.spot_drawdown is not None or self.spot_rally is not None:
            ref_spot = self._require("spot", reference, "reference")
            spot = self._require("spot", market_data, "current")
            if self.spot_drawdown is not None:
                if spot > ref_spot * (1.0 - self.spot_drawdown):
                    return False
            if self.spot_rally is not None:
                if spot < ref_spot * (1.0 + self.spot_rally):
                    return False

        if self.vol_increase is not None or self.vol_decrease is not None:
            ref_vol = self._require("volatility", reference, "reference")
            vol = self._require("volatility", market_data, "current")
            if self.vol_increase is not None:
                if vol < ref_vol + self.vol_increase:
                    return False
            if self.vol_decrease is not None:
                if vol > ref_vol - self.vol_decrease:
                    return False

        return True


class TriggeredHedgeStrategy(MultiGreekHedgeStrategy):
    """
    Greek hedging that activates only after a realized market move.

    Until a trigger fires, should_hedge is always False regardless of how
    far the Greeks drift. Once armed (latched by default), the inherited
    multi-Greek threshold logic applies: hedge whenever any target Greek
    deviates beyond its threshold, sized by the joint solve.

    Attributes:
        triggers: Trigger conditions; ANY firing arms the strategy
        latch: If True (default), the strategy stays armed once triggered
            even if the market recovers; if False, arming follows the
            current trigger state each step
    """

    def __init__(
        self,
        triggers: List[HedgeTrigger],
        name: str = "TriggeredHedge",
        targets: Optional[List[HedgeTarget]] = None,
        hedge_instruments: Optional[List[BaseHedgeInstrument]] = None,
        latch: bool = True,
        rebalance_frequency: str = "continuous",
        min_time_between_hedges: Optional[timedelta] = None,
        instrument_costs: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize trigger-armed hedging strategy.

        Args:
            triggers: Trigger conditions (unique names)
            name: Strategy name
            targets: Greeks to control once armed (default: delta to 0)
            hedge_instruments: Instruments to trade (default: spot only)
            latch: Stay armed after the first firing (default True)
            rebalance_frequency: Frequency gating once armed
            min_time_between_hedges: Minimum time between hedges
            instrument_costs: Optional cost weight per instrument name

        Raises:
            ValidationError: If parameters are invalid
        """
        if not triggers:
            raise ValidationError("At least one HedgeTrigger is required")
        trigger_names = [t.name for t in triggers]
        if len(set(trigger_names)) != len(trigger_names):
            raise ValidationError(f"Duplicate trigger names: {trigger_names}")

        if targets is None:
            targets = [HedgeTarget("delta")]
        if hedge_instruments is None:
            hedge_instruments = [SpotHedgeInstrument()]

        super().__init__(
            name=name,
            targets=targets,
            hedge_instruments=hedge_instruments,
            rebalance_frequency=rebalance_frequency,
            min_time_between_hedges=min_time_between_hedges,
            instrument_costs=instrument_costs,
        )

        self.triggers = list(triggers)
        self.latch = latch

        self._reference_market: Optional[Dict[str, float]] = None
        self._armed = False
        self._fired_triggers: Dict[str, datetime] = {}

    @property
    def armed(self) -> bool:
        """Whether a trigger has fired (and, if unlatched, still holds)."""
        return self._armed

    def on_step(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ):
        """Capture the trade-date reference and evaluate triggers."""
        if self._reference_market is None:
            self._reference_market = dict(market_data)
            return

        any_met = False
        for trigger in self.triggers:
            if trigger.is_met(self._reference_market, market_data):
                any_met = True
                if trigger.name not in self._fired_triggers:
                    self._fired_triggers[trigger.name] = current_time

        if self.latch:
            self._armed = self._armed or any_met
        else:
            self._armed = any_met

    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
    ) -> bool:
        """Hedge only when armed, then per inherited threshold logic."""
        if not self._armed:
            return False
        return super().should_hedge(
            current_time, portfolio_greeks, market_data, **kwargs
        )

    def get_parameters(self) -> Dict[str, object]:
        """Get strategy parameters."""
        params = super().get_parameters()
        params.update(
            {
                "triggers": [
                    {
                        "name": t.name,
                        "spot_drawdown": t.spot_drawdown,
                        "spot_rally": t.spot_rally,
                        "vol_increase": t.vol_increase,
                        "vol_decrease": t.vol_decrease,
                    }
                    for t in self.triggers
                ],
                "latch": self.latch,
            }
        )
        return params

    def get_statistics(self) -> Dict[str, object]:
        """Get strategy statistics."""
        stats = super().get_statistics()
        stats.update(
            {
                "armed": self._armed,
                "fired_triggers": dict(self._fired_triggers),
                "reference_market": (
                    dict(self._reference_market) if self._reference_market else None
                ),
            }
        )
        return stats

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._reference_market = None
        self._armed = False
        self._fired_triggers = {}

    def __repr__(self) -> str:
        triggers = ", ".join(t.name for t in self.triggers)
        return (
            f"TriggeredHedgeStrategy("
            f"triggers=[{triggers}], armed={self._armed})"
        )
