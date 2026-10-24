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
constant maturity and warn once; run-fixings participants (range accruals,
record Asians) are aged by ``quantark.util.run_fixings`` instead;
float-maturity ``Futures`` hedges are a constant-maturity proxy by design and
never warn.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from quantark.asset.equity.product.option.dcn_option import DCNOption
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.util.exceptions import ValidationError
from quantark.util.run_fixings import participates

from .autocallable import AutocallableLifecycleTracker, lifecycle_clock_origin
from .barrier import TRACKED_BARRIER_PRODUCTS, BarrierLifecycleTracker
from .cashflows import (
    LifecycleCashflowLedger,
    ValuationPoint,
)
from .events import LifecycleEvent, LifecycleEventType
from .ko_reset import tracker_for
from .providers import require_rebased_provider
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


def realized_leg_outcome(tracker, book, events: List[LifecycleEvent], date: pd.Timestamp, env, *, base_date):
    """The realized outcome a terminating autocallable event gives a trade's cash legs (spec 2026-10-23 D12/D13).

    Shared by ``PortfolioLifecycleManager`` and the product replay so both book legs the same way. ``events`` are the
    day's lifecycle events (one terminates the position); ``book`` is the trade's ``LegCashBook``; ``base_date`` is
    the run start (the clock origin when the tracker has none). The terminating event's realized cashflow dates (or
    tracker-clock times) convert into the legs' frame; a knock-out's maturity settlement is the pricing copy's
    terminal payment time from today; a maturity carries the tracker's realized knock-in state.

    Raises:
        CapabilityError: A terminating event other than an autocallable knock-out or maturity.
    """
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
    from quantark.cashleg.booking import RealizedOutcome, run_time
    from quantark.execution.errors import CapabilityError

    terminal = next(event for event in events if event.terminates_position)
    anchor = book.anchor_date
    origin = tracker.clock_origin if tracker.clock_origin is not None else base_date
    today = run_time(anchor, date)

    def run_frame(event_date, event_time):
        # into the legs' frame: dates directly, the tracker's clock times shifted by origin - anchor (0 here)
        if event_date is not None:
            return run_time(anchor, event_date)
        if event_time is not None:
            return float(event_time) + run_time(anchor, origin)
        return None

    cashflow = terminal.realized_cashflow
    determined = run_frame(cashflow.determination_date, cashflow.determination_time) if cashflow else None
    settled = run_frame(cashflow.payment_date, cashflow.payment_time) if cashflow else None
    if terminal.event_type is LifecycleEventType.KNOCK_OUT:
        event_time = today if determined is None else determined
        maturity = today + resolve_terminal_timing(tracker.product_for_pricing(date, env), env).payment_time
        return RealizedOutcome.ko(
            event_time, event_time if settled is None else settled, maturity_settlement_time=maturity
        )
    if terminal.event_type is LifecycleEventType.MATURITY:
        return RealizedOutcome.maturity(tracker.lifecycle.knocked_in, today if settled is None else settled)
    raise CapabilityError(
        f"cash legs: a terminating {terminal.event_type.value} event has no multi-day booking rule"
    )


class PortfolioLifecycleManager:
    """Detects and applies realized lifecycle events for a portfolio replay."""

    def __init__(self, base_date: datetime, *, trade_value: bool = False) -> None:
        self.base_date = pd.Timestamp(base_date).normalize()
        # Trade mode (spec 2026-10-23 portfolio-trade-value D15): positions' cash legs roll and book through
        # one LegCashBook each; their paid cash and receivable PV are part of ``paid_cash`` /
        # ``pending_receivable_pv`` and are also reported on their own.
        self.trade_value = bool(trade_value)
        self.leg_paid_cash: float = 0.0
        self.leg_receivable_pv: float = 0.0
        self._leg_books: Dict[str, Any] = {}
        self._leg_underlyings: Dict[str, str] = {}
        self._leg_receivables: Dict[str, float] = {}
        self.ledger = LifecycleCashflowLedger()
        self.pending_receivable_pv: float = 0.0
        self.paid_cash: float = 0.0
        self._cashflow_underlyings: Dict[str, str] = {}
        # A time-based flow's times run from its tracker's clock origin (the contract's initial_date when set,
        # tracker clock decision A, 2026-10-09), not from base_date: each flow is valued on its own clock.
        self._cashflow_origins: Dict[str, pd.Timestamp] = {}
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

        A tracked contract that matured before ``base_date`` leaves the book here, before the run prices it: its
        tracker warns (``MaturedBeforeRunWarning``) and the run books no cash for it and values it at zero (user
        ruling 2026-10-09).
        """
        matured: List[str] = []
        for position_id, position in list(portfolio.positions.items()):
            product = position.product
            if isinstance(product, DCNOption):
                # W14b: the DCN's own tracker (lifecycle/dcn.py); valued from its pre-close copy at base_date
                tracker = tracker_for(product, quantity=position.quantity, start_date=self.base_date)
                env = deepcopy(portfolio.pricing_environments[position.underlying])
                if tracker.retire_if_matured_before_run(env):
                    matured.append(position_id)
                    continue
                self._autocallable[position_id] = tracker
                position.lifecycle_state = tracker.lifecycle
                position.product = tracker.product_for_pricing(self.base_date, env, initial=True)
            elif isinstance(product, (SnowballOption, PhoenixOption)):
                # A KO-reset (a SnowballOption) gets its own tracker (spec 2026-10-08); an un-knocked
                # REBASED one only prices on Monte Carlo (section 6a).
                require_rebased_provider(
                    product, engines={"pricing": position.engine},
                    consumer="PortfolioLifecycleManager", position=position_id,
                )
                tracker = tracker_for(
                    product,
                    quantity=position.quantity,
                    start_date=self.base_date,
                )
                env = deepcopy(portfolio.pricing_environments[position.underlying])
                if tracker.retire_if_matured_before_run(env):
                    matured.append(position_id)
                    continue
                self._autocallable[position_id] = tracker
                position.lifecycle_state = tracker.lifecycle
                # The position is valued from the tracker's initial copy from the start (review finding I2):
                # the snapped contract, its carried state, and a knocked-in REBASED contract already the
                # ABSOLUTE one it has become; the booked product is not what the run prices.
                position.product = tracker.product_for_pricing(self.base_date, env, initial=True)
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
        for position_id in matured:
            portfolio.remove_position(position_id)

    @property
    def leg_book_ids(self) -> Tuple[str, ...]:
        """Positions with a cash-leg book (trade mode)."""
        return tuple(self._leg_books)

    def register_leg_books(self, portfolio) -> None:
        """Attach a cash-leg book to every position with cash legs (trade mode only; spec 2026-10-23 D15).

        Call once, after ``register_positions`` (if trackers are used) and before the first ``process_day``.
        Leg times run from the contract's clock origin (its ``initial_date`` when set — the tracker's clock —
        else ``base_date``; user ruling 2026-10-23); flows paid before ``base_date`` are history. A position an
        autocallable tracker follows gets its leg times snapped
        with the tracker's rule (D14) and may carry legs that pay on termination; an untracked position may
        carry only legs that pay while it lives (D13).

        Raises:
            CapabilityError: Legs on a position whose termination has no booking rule (barrier-family or DCN
                trackers), or legs the book cannot book.
        """
        if not self.trade_value:
            return
        from quantark.asset.equity.lifecycle.snapping import snap_time
        from quantark.cashleg.booking import LegCashBook, LegClock, refuse_floating_notional
        from quantark.execution.errors import CapabilityError

        for position_id, position in portfolio.positions.items():
            legs = getattr(position, "cash_legs", None)
            if not legs or position_id in self._leg_books:
                continue
            if position_id in self._barrier or isinstance(position.product, DCNOption):
                raise CapabilityError(
                    f"position {position_id}: cash legs on a {type(position.product).__name__} have no "
                    "multi-day booking rule for its termination events"
                )
            refuse_floating_notional(legs, position.product)
            tracked = position_id in self._autocallable
            env = portfolio.pricing_environments[position.underlying]
            origin = (self._autocallable[position_id].clock_origin if tracked
                      else lifecycle_clock_origin(position.product, self.base_date))
            book = LegCashBook(
                legs,
                position._get_unit_notional(env),
                pd.Timestamp(origin).normalize(),
                snap=snap_time if tracked else None,
                terminable=tracked,
                run_start=self.base_date,
            )
            self._leg_books[position_id] = book
            self._leg_underlyings[position_id] = position.underlying
            position.cash_legs = list(book.legs_at_run_start())
            # the book rolls the legs from here on: they are on the valuation clock (user decision 2026-10-24)
            position.cash_legs_clock = LegClock.VALUATION

    def _leg_outcome(self, tracker, book, events: List[LifecycleEvent], date: pd.Timestamp, env):
        """The realized outcome a terminating autocallable event gives the position's cash legs (D12/D13)."""
        return realized_leg_outcome(tracker, book, events, date, env, base_date=self.base_date)

    def _book_legs(self, portfolio, date: pd.Timestamp, terminated_today: Dict[str, Any]) -> None:
        """Roll every live leg book to ``date`` and pay what is due; terminated books pay their receivables."""
        for position_id, book in self._leg_books.items():
            env = portfolio.pricing_environments[self._leg_underlyings[position_id]]
            if position_id in terminated_today:
                day = terminated_today[position_id]
            elif book.terminated:
                day = book.roll(date, env)
            else:
                position = portfolio.positions.get(position_id)
                if position is None:
                    raise ValidationError(
                        f"position {position_id} left the portfolio without a lifecycle termination; "
                        "its cash legs cannot be booked"
                    )
                day = book.roll(date, env)
                position.cash_legs = list(day.live_legs)
            self.leg_paid_cash += day.paid_cash
            self._leg_receivables[position_id] = day.receivable_pv
        self.leg_receivable_pv = float(sum(self._leg_receivables.values()))

    def register_float_rolls(self, portfolio) -> None:
        """Anchor every schedule-free float-maturity position at ``base_date``.

        Call once, before the first ``process_day``, WHETHER OR NOT lifecycle
        trackers are attached: without it these positions would first be seen by
        ``_roll_untracked`` and anchored at the first market date instead, so the
        same book would age differently depending on whether the backtest handles
        lifecycle events (Kimi review 2026-09-03). First sight wins, so calling
        this and ``register_positions`` in either order is idempotent.
        """
        for position_id, position in portfolio.positions.items():
            if is_float_rollable(position.product):
                self._float_roller.register(position_id, position.product, self.base_date)

    def _roll_untracked(self, position_id: str, position, date: pd.Timestamp):
        """Rolled copy of a schedule-free float-maturity product, else None.

        Untracked products with a float maturity but no roll rule are repriced with
        a constant maturity; warn once per position so the choice is visible.
        """
        product = position.product
        if participates(product):
            return None          # aged and fixed by the run's RunFixingRecorder (spec 2026-10-09 run fixings §3.5)
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
        terminated_legs: Dict[str, Any] = {}
        for position_id in list(portfolio.positions.keys()):
            position = portfolio.positions[position_id]
            env = portfolio.pricing_environments[position.underlying]
            spot = float(env.spot)
            if position_id in self._autocallable:
                processed.extend(
                    self._process_autocallable(
                        portfolio, position_id, position, env, spot, date, terminated_legs
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
        self._book_legs(portfolio, date, terminated_legs)
        self._revalue_ledger(portfolio, date)
        # leg cash is part of the day's cash totals (spec 2026-10-23 D15)
        self.paid_cash += self.leg_paid_cash
        self.pending_receivable_pv += self.leg_receivable_pv
        return processed

    def _process_autocallable(
        self, portfolio, position_id, position, env, spot, date, terminated_legs=None
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
            clock_origin=tracker.clock_origin,
        )
        # Capture context BEFORE mutating/removing the position so product_type
        # reflects the product at event time.
        processed = [
            self._processed(position_id, position, event) for event in events
        ]
        if terminated:
            book = self._leg_books.get(position_id)
            if book is not None:
                # decided once, by the outcome, on the day the position ends (spec 2026-10-23 D12)
                outcome = self._leg_outcome(tracker, book, events, date, env)
                terminated_legs[position_id] = book.terminate(outcome, date, env)
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
        if terminated or not tracker.state.alive:
            # a terminal event, or retired as matured before the run (no event, no cash; ruling R31)
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
        clock_origin: Optional[pd.Timestamp] = None,
    ) -> tuple[List[LifecycleEvent], bool]:
        """Register determined cashflows and namespace their portfolio IDs.

        ``clock_origin`` is the day the tracker's time-based flows are measured from (``base_date`` when omitted).
        """
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
                self._cashflow_origins[global_id] = (
                    self.base_date if clock_origin is None else pd.Timestamp(clock_origin).normalize()
                )
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
        day = pd.Timestamp(date).normalize()
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
                origin = self._cashflow_origins.get(cashflow.cashflow_id, self.base_date)
                point = ValuationPoint(time=max(0.0, (day - origin).days / 365.0))
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
