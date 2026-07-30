"""
Lifecycle event management for dynamic scenario simulations.

``LifecycleManager`` is a thin dynamic-scenario adapter over the shared
:class:`~quantark.asset.equity.lifecycle.PortfolioLifecycleManager`. The core
manager does all the work — attaching trackers, observing each day's close,
mutating/removing positions, and valuing determined cashflows. This adapter
only maps the neutral ``ProcessedLifecycleEvent`` records the core returns into
the dynamic-scenario :class:`LifecycleEventSnapshot` result type.

Settlement convention: determination removes the contingent claim, pending
cashflows remain at discounted value until payment, and only then move to paid
cash (``portfolio_value = live MTM + pending PV + paid cash``).
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
        """Backward-compatible alias for cash that has actually been paid."""
        return self._core.paid_cash

    @property
    def pending_receivable_pv(self) -> float:
        """PV of determined lifecycle cashflows awaiting payment."""
        return self._core.pending_receivable_pv

    @property
    def paid_cash(self) -> float:
        """Cumulative lifecycle cash paid through the current day."""
        return self._core.paid_cash

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
        cashflow = event.realized_cashflow
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
            cashflow_id=(
                cashflow.cashflow_id if cashflow is not None else None
            ),
            determination_date=(
                cashflow.determination_date if cashflow is not None else None
            ),
            determination_time=(
                cashflow.determination_time if cashflow is not None else None
            ),
            payment_date=(
                cashflow.payment_date if cashflow is not None else None
            ),
            payment_time=(
                cashflow.payment_time if cashflow is not None else None
            ),
        )
