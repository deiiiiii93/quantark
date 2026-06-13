"""
Lifecycle event management for dynamic scenario simulations.

``LifecycleManager`` is a thin dynamic-scenario adapter over the shared
:class:`~quantark.asset.equity.lifecycle.PortfolioLifecycleManager`. The core
manager does all the work — attaching trackers, observing each day's close,
mutating/removing positions, and accumulating settlement cash. This adapter
only maps the neutral ``ProcessedLifecycleEvent`` records the core returns into
the dynamic-scenario :class:`LifecycleEventSnapshot` result type.

Settlement convention: terminated positions settle to cash that remains part
of portfolio value (``portfolio_value = positions MTM + realized_cash``), so
daily P&L is continuous across event days. Ledger cash earns no interest
within the path.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pandas as pd

from quantark.asset.equity.lifecycle import (
    PortfolioLifecycleManager,
    ProcessedLifecycleEvent,
)
from quantark.dynamicscenario.results.dynamic_results import LifecycleEventSnapshot


class LifecycleManager:
    """Detects and applies lifecycle events for a dynamic scenario run.

    Wraps a :class:`PortfolioLifecycleManager` and converts the events it
    produces into dynamic-scenario :class:`LifecycleEventSnapshot` records.
    """

    def __init__(self, base_date: datetime) -> None:
        self._core = PortfolioLifecycleManager(base_date=base_date)

    @property
    def base_date(self) -> pd.Timestamp:
        return self._core.base_date

    @property
    def realized_cash(self) -> float:
        """Cumulative settlement cash booked across all processed days."""
        return self._core.realized_cash

    @property
    def num_tracked(self) -> int:
        """Number of positions with an attached lifecycle tracker."""
        return self._core.num_tracked

    def register_positions(self, portfolio) -> None:
        """Attach lifecycle trackers to all trackable positions."""
        self._core.register_positions(portfolio)

    def date_for_day(
        self, day_index: int, day_date: Optional[datetime]
    ) -> pd.Timestamp:
        """Resolve the calendar date of a simulation day."""
        return self._core.date_for_day(day_index, day_date)

    def process_day(
        self, portfolio, day_index: int, day_date: Optional[datetime]
    ) -> List[LifecycleEventSnapshot]:
        """Observe this day's close and return per-event snapshots."""
        processed = self._core.process_day(portfolio, day_index, day_date)
        return [self._to_snapshot(item) for item in processed]

    @staticmethod
    def _to_snapshot(item: ProcessedLifecycleEvent) -> LifecycleEventSnapshot:
        event = item.event
        event_date = event.date
        if isinstance(event_date, pd.Timestamp):
            event_date = event_date.to_pydatetime()
        return LifecycleEventSnapshot(
            position_id=item.position_id,
            underlying=item.underlying,
            product_type=item.product_type,
            event_type=event.event_type.value,
            date=event_date,
            observation_index=event.observation_index,
            spot=event.spot,
            barrier=event.barrier,
            payoff=event.payoff,
            cashflow=event.cashflow,
            terminates_position=event.terminates_position,
        )
