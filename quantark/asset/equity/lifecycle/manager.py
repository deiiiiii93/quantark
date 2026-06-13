"""
Portfolio-driving lifecycle manager shared by historical backtests
(``quantark.backtest.equity``) and dynamic scenario simulation
(``quantark.dynamicscenario``).

``PortfolioLifecycleManager`` scans a working portfolio for products with
realized lifecycle semantics, attaches the appropriate tracker from this
package, and processes each day's close: detecting events, mutating positions
(KI flags / product substitution / engine override), removing terminated
positions, and accumulating settlement cash in a run-level ledger.

It is intentionally consumer-agnostic. ``process_day`` returns a list of
``ProcessedLifecycleEvent`` records — each pairing the raw
:class:`LifecycleEvent` with the position context captured *before* mutation
(``position_id``, ``underlying``, ``product_type``). Consumers convert these
into their own native records (a dynamic-scenario snapshot, a backtest event
row, ...).

The manager never imports ``quantark.portfolio``; it operates structurally on
any object exposing ``positions`` (mapping of position-id to objects with
``product``/``quantity``/``underlying``/``engine``), ``pricing_environments``,
and ``remove_position``. Keeping it duck-typed avoids an
``asset -> portfolio -> asset`` import cycle.

Settlement convention: terminated positions settle to cash that remains part
of portfolio value (``portfolio_value = positions MTM + realized_cash``), so
daily P&L is continuous across event days. Ledger cash earns no interest
within the path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_option import SnowballOption

from .autocallable import AutocallableLifecycleTracker
from .barrier import TRACKED_BARRIER_PRODUCTS, BarrierLifecycleTracker
from .events import LifecycleEvent


@dataclass(frozen=True)
class ProcessedLifecycleEvent:
    """One realized lifecycle event plus the position context at event time.

    The position context (``underlying``, ``product_type``) is captured before
    the manager mutates or removes the position, so it always reflects the
    product as it stood when the event fired.

    Attributes:
        position_id: Identifier of the position the event applies to.
        underlying: Underlying asset name of the position.
        product_type: Class name of the product at event time.
        event: The raw :class:`LifecycleEvent` produced by the tracker.
    """

    position_id: str
    underlying: str
    product_type: str
    event: LifecycleEvent


class PortfolioLifecycleManager:
    """Detects and applies realized lifecycle events for a portfolio replay."""

    def __init__(self, base_date: datetime) -> None:
        self.base_date = pd.Timestamp(base_date).normalize()
        self.realized_cash: float = 0.0
        self._autocallable: Dict[str, AutocallableLifecycleTracker] = {}
        self._barrier: Dict[str, BarrierLifecycleTracker] = {}

    @property
    def num_tracked(self) -> int:
        """Number of positions with an attached lifecycle tracker."""
        return len(self._autocallable) + len(self._barrier)

    def register_positions(self, portfolio) -> None:
        """
        Attach lifecycle trackers to all trackable positions.

        Call once, after portfolio construction and before the first
        ``process_day`` call; calling it again would re-register trackers
        with fresh (reset) lifecycle state. Positions added later (e.g.
        hedge instruments) are intentionally untracked.
        """
        for position_id, position in portfolio.positions.items():
            product = position.product
            if isinstance(product, KnockOutResetSnowballOption):
                warnings.warn(
                    "KO-reset snowball lifecycle (barrier reset on KO) is not "
                    f"tracked; position {position_id} will be repriced as of "
                    "today on every day.",
                    UserWarning,
                )
                continue
            if isinstance(product, (SnowballOption, PhoenixOption)):
                self._autocallable[position_id] = AutocallableLifecycleTracker(
                    product=product,
                    quantity=position.quantity,
                    start_date=self.base_date,
                )
            elif isinstance(product, TRACKED_BARRIER_PRODUCTS):
                self._barrier[position_id] = BarrierLifecycleTracker(
                    product=product,
                    quantity=position.quantity,
                    start_date=self.base_date,
                )

    def date_for_day(self, day_index: int, day_date: Optional[datetime]) -> pd.Timestamp:
        """Resolve the calendar date of a replay day."""
        if day_date is not None:
            return pd.Timestamp(day_date).normalize()
        return self.base_date + pd.Timedelta(days=day_index)

    def process_day(
        self, portfolio, day_index: int, day_date: Optional[datetime]
    ) -> List[ProcessedLifecycleEvent]:
        """Observe this day's close for every tracked position.

        Returns the events that fired today, each carrying the position
        context captured before any mutation/removal.
        """
        date = self.date_for_day(day_index, day_date)
        processed: List[ProcessedLifecycleEvent] = []
        for position_id in list(portfolio.positions.keys()):
            position = portfolio.positions[position_id]
            env = portfolio.pricing_environments[position.underlying]
            spot = float(env.spot)
            if position_id in self._autocallable:
                processed.extend(
                    self._process_autocallable(
                        portfolio, position_id, position, env, spot, date
                    )
                )
            elif position_id in self._barrier:
                processed.extend(
                    self._process_barrier(
                        portfolio, position_id, position, env, spot, date
                    )
                )
        return processed

    def _process_autocallable(
        self, portfolio, position_id, position, env, spot, date
    ) -> List[ProcessedLifecycleEvent]:
        tracker = self._autocallable[position_id]
        lifecycle_product = tracker.product_for_lifecycle()
        events = tracker.observe(date, lifecycle_product, env, spot)
        maturity_event = tracker.settle_maturity_if_due(
            date, lifecycle_product, env, spot
        )
        if maturity_event is not None:
            events.append(maturity_event)

        # Capture context BEFORE mutating/removing the position so product_type
        # reflects the product at event time.
        processed = [
            self._processed(position_id, position, event) for event in events
        ]
        if self._book_events(events):
            portfolio.remove_position(position_id)
            del self._autocallable[position_id]
        else:
            position.product = tracker.product_for_pricing(date, env)
        return processed

    def _process_barrier(
        self, portfolio, position_id, position, env, spot, date
    ) -> List[ProcessedLifecycleEvent]:
        tracker = self._barrier[position_id]
        events = tracker.observe(date, env, spot)

        # Capture context BEFORE mutating/removing the position.
        processed = [
            self._processed(position_id, position, event) for event in events
        ]
        if self._book_events(events):
            portfolio.remove_position(position_id)
            del self._barrier[position_id]
        else:
            position.product = tracker.product_for_pricing(date, env)
            engine_override = tracker.engine_for_pricing()
            if engine_override is not None:
                position.engine = engine_override
        return processed

    def _book_events(self, events: List[LifecycleEvent]) -> bool:
        """Book event cashflows to the ledger; return True if terminated."""
        terminated = False
        for event in events:
            self.realized_cash += event.cashflow
            terminated = terminated or event.terminates_position
        return terminated

    @staticmethod
    def _processed(
        position_id, position, event: LifecycleEvent
    ) -> ProcessedLifecycleEvent:
        return ProcessedLifecycleEvent(
            position_id=position_id,
            underlying=position.underlying,
            product_type=type(position.product).__name__,
            event=event,
        )
