"""
Portfolio-driving lifecycle manager shared by historical backtests
(``quantark.backtest.equity``) and dynamic scenario simulation
(``quantark.dynamicscenario``).

``PortfolioLifecycleManager`` scans a working portfolio for products with
realized lifecycle semantics, attaches the appropriate tracker from this
package, and processes each day's close: detecting events, mutating positions
(KI flags / product substitution / engine override), removing terminated
positions, and retaining determined cashflows in a run-level ledger.

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

Settlement convention: determination removes the contingent claim, a fixed
receivable remains until payment, and paid cash then remains in portfolio
value (``live MTM + pending PV + paid cash``). Paid ledger cash earns no
interest within the path.

The manager also ages untracked *schedule-free* float-maturity contracts
(``float_roll.FLOAT_ROLLABLE_PRODUCTS``) by ``days / 365`` from the day it
first sees the position, so a ``maturity=1.0`` vanilla is not repriced as a
one-year option on every day of the run. Untracked products with a float
maturity and no roll rule (schedule-bearing contracts) are repriced with a
constant maturity and warn once; float-maturity ``Futures`` hedges are a
constant-maturity proxy by design and never warn.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
)
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_option import SnowballOption

from .autocallable import AutocallableLifecycleTracker
from .barrier import TRACKED_BARRIER_PRODUCTS, BarrierLifecycleTracker
from .cashflows import (
    LifecycleCashflowLedger,
    ValuationPoint,
)
from .events import LifecycleEvent
from .float_roll import FloatMaturityRoller, has_unrolled_float_maturity, is_float_rollable


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
        self.ledger = LifecycleCashflowLedger()
        self.pending_receivable_pv: float = 0.0
        self.paid_cash: float = 0.0
        self._cashflow_underlyings: Dict[str, str] = {}
        self._autocallable: Dict[str, AutocallableLifecycleTracker] = {}
        self._barrier: Dict[str, BarrierLifecycleTracker] = {}
        # Untracked schedule-free float-maturity contracts (patch spec 2026-09-03 §4)
        self._float_roller = FloatMaturityRoller()
        self._unrolled_warned: set = set()

    @property
    def realized_cash(self) -> float:
        """Backward-compatible alias for cash that has actually been paid."""
        return self.paid_cash

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
        hedge instruments) are intentionally untracked. Schedule-free
        float-maturity products present here have their roll measured from
        ``base_date`` (the trackers' convention); others roll from the first
        day the manager sees them.
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
                tracker = AutocallableLifecycleTracker(
                    product=product,
                    quantity=position.quantity,
                    start_date=self.base_date,
                )
                self._autocallable[position_id] = tracker
                position.lifecycle_state = tracker.lifecycle
            elif isinstance(product, TRACKED_BARRIER_PRODUCTS):
                tracker = BarrierLifecycleTracker(
                    product=product,
                    quantity=position.quantity,
                    start_date=self.base_date,
                )
                self._barrier[position_id] = tracker
                position.lifecycle_state = tracker.state
            elif is_float_rollable(product):
                self._float_roller.register(position_id, product, self.base_date)


    def _roll_untracked(self, position_id: str, position, date: pd.Timestamp):
        """Rolled copy of a schedule-free float-maturity product, else None.

        Untracked products with a float maturity but no roll rule are repriced with
        a constant maturity; warn once per position so the choice is visible.
        """
        product = position.product
        if not is_float_rollable(product):
            if has_unrolled_float_maturity(product) and position_id not in self._unrolled_warned:
                self._unrolled_warned.add(position_id)
                warnings.warn(
                    f"position {position_id} ({type(product).__name__}) has a float maturity and "
                    "no roll rule; it is repriced with a constant maturity every day",
                    UserWarning,
                )
            return None
        self._float_roller.register(position_id, product, date)
        return self._float_roller.rolled(position_id, product, date)

    def pricing_products(self, portfolio, date) -> Dict[str, Any]:
        """Per-position pricing product for ``date`` under the CURRENT lifecycle state.

        Pure accessor for the PnL explain recorder: tracked positions return
        ``tracker.product_for_pricing(date, env)`` (the alive contract rolled
        to ``date``; call it before ``process_day`` to get the pre-event
        contract); untracked schedule-free float-maturity products return a
        fresh copy rolled to ``date``; other untracked positions return their
        current product. No position is mutated (the roller only records a
        position's first-sight base).
        """
        date = pd.Timestamp(date).normalize()
        out: Dict[str, Any] = {}
        for position_id, position in portfolio.positions.items():
            env = portfolio.pricing_environments[position.underlying]
            if position_id in self._autocallable:
                out[position_id] = self._autocallable[position_id].product_for_pricing(date, env)
            elif position_id in self._barrier:
                out[position_id] = self._barrier[position_id].product_for_pricing(date, env)
            else:
                rolled = self._roll_untracked(position_id, position, date)
                out[position_id] = position.product if rolled is None else rolled
        return out

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
            else:
                rolled = self._roll_untracked(position_id, position, date)
                if rolled is not None:
                    position.product = rolled
        self._float_roller.retain(portfolio.positions.keys())
        self._revalue_ledger(portfolio, date)
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

        events, terminated = self._book_events(
            position_id,
            position.underlying,
            events,
        )
        # Capture context BEFORE mutating/removing the position so product_type
        # reflects the product at event time.
        processed = [
            self._processed(position_id, position, event) for event in events
        ]
        if terminated:
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

        events, terminated = self._book_events(
            position_id,
            position.underlying,
            events,
        )
        # Capture context BEFORE mutating/removing the position.
        processed = [
            self._processed(position_id, position, event) for event in events
        ]
        if terminated:
            portfolio.remove_position(position_id)
            del self._barrier[position_id]
        else:
            position.product = tracker.product_for_pricing(date, env)
            engine_override = tracker.engine_for_pricing()
            if engine_override is not None:
                position.engine = engine_override
        return processed

    def _book_events(
        self,
        position_id: str,
        underlying: str,
        events: List[LifecycleEvent],
    ) -> tuple[List[LifecycleEvent], bool]:
        """Register determined cashflows and namespace their portfolio IDs."""
        managed_events: List[LifecycleEvent] = []
        terminated = False
        for event in events:
            cashflow = event.realized_cashflow
            if cashflow is not None:
                global_id = f"{position_id}:{cashflow.cashflow_id}"
                metadata = dict(cashflow.metadata)
                metadata.update(
                    {
                        "position_id": position_id,
                        "underlying": underlying,
                    }
                )
                ledger_cashflow = replace(
                    cashflow,
                    cashflow_id=global_id,
                    metadata=metadata,
                )
                self.ledger.register(ledger_cashflow)
                self._cashflow_underlyings[global_id] = underlying
                event = replace(
                    event,
                    realized_cashflow=ledger_cashflow,
                )
            managed_events.append(event)
            terminated = terminated or event.terminates_position
        return managed_events, terminated

    def _revalue_ledger(self, portfolio, date: pd.Timestamp) -> None:
        pending_pv = 0.0
        paid_cash = 0.0
        elapsed = max(
            0.0,
            (
                pd.Timestamp(date).normalize() - self.base_date
            ).days
            / 365.0,
        )
        for cashflow in self.ledger.cashflows:
            underlying = self._cashflow_underlyings[cashflow.cashflow_id]
            env = portfolio.pricing_environments[underlying]
            if cashflow.payment_date is not None:
                point = ValuationPoint(
                    date=pd.Timestamp(date).to_pydatetime()
                )
                valuation_env = env
                if env.valuation_date != point.date:
                    valuation_env = deepcopy(env)
                    valuation_env.valuation_date = point.date
            else:
                point = ValuationPoint(time=elapsed)
                valuation_env = env

            single = LifecycleCashflowLedger([cashflow])
            if single.pending(point):
                pending_pv += single.pending_pv(point, valuation_env)
            else:
                paid_cash += cashflow.amount

        self.pending_receivable_pv = float(pending_pv)
        self.paid_cash = float(paid_cash)

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
