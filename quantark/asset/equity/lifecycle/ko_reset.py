"""Lifecycle tracking for the knock-out-reset snowball (spec 2026-10-08 KO-reset lifecycle).

The contract's rules, as the desk stated them and every engine applies them:

* R1 a knock-in during schedule 1 replaces it by schedule 2 for good, schedule 2 in force from the knock-in
  observation, that observation included;
* R2 a contract not knocked in when schedule 1 ends has matured there: it pays V0 on schedule 1's contract
  length, and a knock-in is never tested after that;
* R3 a schedule-1 knock-out and a knock-in at the same observation: the knock-out wins;
* R4 a carried knock-in happened.

A knocked-in REBASED contract is the ABSOLUTE contract ``with_absolute_schedule_2`` builds at the knock-in
time, so every engine prices an ordinary knocked-in contract and the tracker books from that same contract.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing_at
from quantark.asset.equity.product.option.accrual import knock_in_anchor_for
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.util.exceptions import ValidationError

from .autocallable import AutocallableLifecycleTracker, CalendarSchedule, DateResolver
from .events import LifecycleEvent, LifecycleEventType
from .state import AutocallableLifecycleState

DAYS_PER_YEAR = 365.0

__all__ = ["KOResetCalendarSchedule", "KOResetLifecycleTracker", "SchedulePlacement", "tracker_for"]


@dataclass(frozen=True)
class SchedulePlacement:
    """Schedule 2 and the knocked-in maturity on a run calendar: an ABSOLUTE contract's, or a REBASED one's for one
    knock-in day. Every ``*_day`` is a position in the calendar, -1 past its end."""

    ko_due_day: np.ndarray
    ko_barrier: np.ndarray
    ko_payoff: np.ndarray
    ko_settlement_day: np.ndarray
    ko_settlement_delay: np.ndarray
    final_due_day: int
    final_settlement_day: int
    final_settlement_delay: float
    final_tenor: float


@dataclass(frozen=True)
class KOResetCalendarSchedule(CalendarSchedule):
    """A KO-reset's observations on a run calendar (``KOResetLifecycleTracker.resolve_calendar_schedule``).

    The inherited ``ko_*`` fields are schedule 1 and the ``ki_*`` fields the knock-in; ``terminal_*`` is the
    latest terminal the calendar may have to hold (the knocked-in maturity of the last possible knock-in), which
    is what the run's data-end check reads. ``pre_*`` is schedule 1's maturity (V0 on ``pre_tenor``). Schedule 2
    is ``absolute`` for an ABSOLUTE contract, or ``place(knock_in_day)`` for a REBASED one.
    """

    pre_due_day: int
    pre_settlement_day: int
    pre_settlement_delay: float
    pre_tenor: float
    rebased: bool
    absolute: Optional[SchedulePlacement]
    place: Optional[Callable[[int], SchedulePlacement]]

    def placement(self, knock_in_day: int) -> SchedulePlacement:
        """Schedule 2 for a path knocked in on ``knock_in_day``."""
        return self.place(int(knock_in_day)) if self.rebased else self.absolute


class KOResetLifecycleTracker(AutocallableLifecycleTracker):
    """Tracks one ``KnockOutResetSnowballOption`` under rules R1-R4 (see the module docstring).

    The knock-in close is the state's ``ki_date`` (set by ``mark_ki``); on the numeric clock the knock-in time
    is ``(ki_date - start_date).days / 365``. It is the reset's TIMING. The reset's ACCRUAL hangs from the state's
    ``knock_in_anchor``: the knock-in record that fired, at its contractual fixing, resolved once at the knock-in
    (accrual time spec §5.1). Without a ``lifecycle`` the starting state is the product's carried one:
    ``_otc_lifecycle_knocked_in``, ``_otc_lifecycle_knock_in_date`` and ``_otc_lifecycle_knock_in_anchor``.
    """

    def __init__(
        self,
        *,
        product: Any,
        quantity: float,
        lifecycle: Optional[AutocallableLifecycleState] = None,
        start_date: Optional[pd.Timestamp] = None,
        date_resolver: Optional[DateResolver] = None,
        has_lifecycle: bool = True,
    ) -> None:
        if lifecycle is None:
            knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
            lifecycle = AutocallableLifecycleState(knocked_in=knocked_in)
            if knocked_in:
                lifecycle.ki_date = getattr(product, "_otc_lifecycle_knock_in_date", None)
                lifecycle.knock_in_anchor = getattr(product, "_otc_lifecycle_knock_in_anchor", None)
        if lifecycle.knocked_in and product.post_ko_mode == PostKOScheduleMode.REBASED:
            if lifecycle.ki_date is None:
                raise ValidationError(
                    "KO-reset snowball: a carried knock-in on a REBASED contract needs its knock-in date "
                    "(_otc_lifecycle_knock_in_date): schedule 2 holds offsets from it and cannot be placed without it"
                )
            if lifecycle.knock_in_anchor is None and product.knock_in_anchor_required():
                # accrual time spec §5.1: resolved once at the knock-in and persisted; a date can match zero
                # records or several, so it is never looked up by date
                raise ValidationError(
                    "KO-reset snowball: a carried knock-in on a REBASED contract needs its knock-in anchor "
                    "(_otc_lifecycle_knock_in_anchor, or knock_in_anchor on the state): schedule 2's accrual hangs "
                    "from the record that fired, which is never looked up by date"
                )
        super().__init__(
            product=product,
            quantity=quantity,
            lifecycle=lifecycle,
            start_date=start_date,
            date_resolver=date_resolver,
            has_lifecycle=has_lifecycle,
        )
        self._previous_close: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # The contract in force
    # ------------------------------------------------------------------

    def knock_in_time(self) -> Optional[float]:
        """Years from ``start_date`` to the knock-in close (numeric clock); ``None`` while not knocked in."""
        if not self.lifecycle.knocked_in or self.lifecycle.ki_date is None or self.start_date is None:
            return None
        days = (
            pd.Timestamp(self.lifecycle.ki_date).normalize() - pd.Timestamp(self.start_date).normalize()
        ).days
        return days / DAYS_PER_YEAR

    def contract(self) -> Any:
        """The snapped contract, or the ABSOLUTE contract a knocked-in REBASED one has become."""
        if self.lifecycle.knocked_in and self.product.post_ko_mode == PostKOScheduleMode.REBASED:
            knock_in_time = self.knock_in_time()
            if knock_in_time is None:
                raise ValidationError("a knocked-in REBASED KO-reset needs start_date and its knock-in date")
            return self.product.with_absolute_schedule_2(knock_in_time, anchor=self.lifecycle.knock_in_anchor)
        return self.product

    def product_for_lifecycle(self) -> Any:
        product = super().product_for_lifecycle()
        setattr(product, "_otc_lifecycle_knock_in_date", self.lifecycle.ki_date)
        return product

    def _contract_for_pricing(self) -> Any:
        product = deepcopy(self.contract())
        setattr(product, "_otc_lifecycle_knocked_in", self.lifecycle.knocked_in)
        setattr(product, "_otc_lifecycle_knock_in_date", self.lifecycle.ki_date)
        setattr(product, "_otc_lifecycle_knock_in_anchor", self.lifecycle.knock_in_anchor)
        return product

    def _age_extra(self, product, elapsed, date, pricing_env, keep_landing) -> None:
        """Age schedule 2 with schedule 1; a REBASED one (offsets) stays until a knock-in converts it."""
        if product.post_ko_mode == PostKOScheduleMode.ABSOLUTE:
            product.post_barrier_config, _ = product.post_barrier_config.time_shift(
                elapsed, date, pricing_env, keep_landing=keep_landing
            )

    # ------------------------------------------------------------------
    # Records, tenors and terminal timings
    # ------------------------------------------------------------------

    def _schedule_records(self, product: Any, env: PricingEnvironment, config: Any) -> List[Dict[str, Any]]:
        """One schedule's knock-out records; the cash is the product's own ``_ko_cash`` (both schedules)."""
        schedule_env = self._schedule_resolution_env(product, env)
        resolved, rates, sources = product._resolve_ko_schedule(config, schedule_env)
        payoffs = [
            product._ko_cash(record.observation_time, rate, source, schedule_env)
            for record, rate, source in zip(resolved, rates, sources)
        ]
        return self._record_dicts(product, resolved, payoffs)

    def pre_contract_tenor(self, env: PricingEnvironment) -> float:
        """Schedule 1's contract length (V0's tenor)."""
        return float(self.product._resolve_pre_contract_tenor(self._schedule_resolution_env(self.product, env)))

    def final_contract_tenor(self, env: PricingEnvironment) -> float:
        """The knocked-in contract's length (V1's tenor)."""
        contract = self.contract()
        return float(contract._resolve_post_contract_tenor(self._schedule_resolution_env(contract, env)))

    def _last_pre_record(self, product: Any, env: PricingEnvironment) -> Dict[str, Any]:
        records = self._schedule_records(product, env, product.barrier_config)
        if not records:
            raise ValidationError(
                "KO-reset snowball: schedule 1 has no observation left, so a contract not knocked in has matured "
                "before this run; it cannot be tracked from here"
            )
        return records[-1]

    def _pre_timing(self, product: Any, env: PricingEnvironment, last: Dict[str, Any]):
        """V0's terminal timing: determined at schedule 1's last observation."""
        schedule_env = self._schedule_resolution_env(product, env)
        if self._uses_date_timing(product) and last["determination_date"] is not None:
            return resolve_terminal_timing_at(
                product, schedule_env, determination_date=pd.Timestamp(last["determination_date"]).to_pydatetime(),
                cashflow_id="ko_reset_v0_terminal",
            )
        return resolve_terminal_timing_at(
            product, schedule_env, determination_time=float(last["time"]), cashflow_id="ko_reset_v0_terminal"
        )

    def _final_timing(self, env: PricingEnvironment):
        """V1's terminal timing: determined at the knocked-in contract's end."""
        contract = self.contract()
        schedule_env = self._schedule_resolution_env(contract, env)
        end_date = _last_observation_date(contract.post_barrier_config.ko_observation_schedule)
        if end_date is not None and self._uses_date_timing(contract):
            return resolve_terminal_timing_at(
                contract, schedule_env, determination_date=end_date, cashflow_id="ko_reset_v1_terminal"
            )
        return resolve_terminal_timing_at(
            contract, schedule_env, determination_time=float(contract.get_post_maturity_time(schedule_env)),
            cashflow_id="ko_reset_v1_terminal",
        )

    def _due_at_previous_close(self, product: Any, record: Dict[str, Any]) -> bool:
        """Whether ``record`` was already due at the previous observed close (none: nothing was)."""
        if self._previous_close is None:
            return False
        previous = self._previous_close
        return self._record_is_due(previous, self._valuation_point(previous, product), record)

    # ------------------------------------------------------------------
    # The day's close
    # ------------------------------------------------------------------

    def observe(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment, spot: float
    ) -> List[LifecycleEvent]:
        """Schedule-1 knock-out, then the knock-in, then schedule 2 (the knock-in close included)."""
        valuation_point = self._valuation_point(date, product)
        self.lifecycle.valuation_point = valuation_point
        if not self.lifecycle.alive or not self.has_lifecycle:
            return []
        events: List[LifecycleEvent] = []
        try:
            if not self.lifecycle.knocked_in:
                knock_out = self._test_knock_outs(date, product, env, spot, valuation_point, schedule=1)
                if knock_out is not None:
                    return [knock_out]                      # R3: a knock-out ends the day
                knock_in = self._test_knock_in(date, product, env, spot, valuation_point)
                if knock_in is None:
                    return events
                events.append(knock_in)
                self._consume_schedule_two(env)
            contract = self.contract()
            if not contract.barrier_config.disable_ko_after_ki:
                knock_out = self._test_knock_outs(date, contract, env, spot, valuation_point, schedule=2)
                if knock_out is not None:
                    events.append(knock_out)
            return events
        finally:
            self._previous_close = pd.Timestamp(date).normalize()

    def _test_knock_outs(self, date, product, env, spot, valuation_point, *, schedule: int):
        config = product.barrier_config if schedule == 1 else product.post_barrier_config
        observed = (
            self.lifecycle.observed_ko_indices if schedule == 1 else self.lifecycle.observed_post_ko_indices
        )
        for idx, rec in enumerate(self._schedule_records(product, env, config)):
            if idx in observed or not self._record_is_due(date, valuation_point, rec):
                continue
            observed.add(idx)
            if not self._barrier_hit(spot, rec["barrier"], product.is_reverse, is_ko=True):
                continue
            before = self._state_snapshot()
            payoff = float(rec["payoff"])
            cashflow = self.quantity * payoff
            settlement_date = rec.get("settlement_date")
            if settlement_date is not None:
                settlement_date = pd.Timestamp(settlement_date).to_pydatetime()
            realized = self._record_cashflow(
                LifecycleEventType.KNOCK_OUT, f"knock-out:ko{schedule}:{idx}", cashflow, rec, valuation_point,
                metadata={"schedule": schedule},
            )
            if self.lifecycle.mark_ko(
                pd.Timestamp(date).to_pydatetime(),
                cashflow,
                settlement_date=settlement_date,
                realized_cashflow=realized,
                valuation_point=valuation_point,
            ):
                return LifecycleEvent(
                    event_type=LifecycleEventType.KNOCK_OUT,
                    date=date,
                    spot=spot,
                    observation_index=idx,
                    barrier=rec["barrier"],
                    payoff=payoff,
                    cashflow=cashflow,
                    realized_cashflow=realized,
                    terminates_position=True,
                    state_before=before,
                    state_after=self._state_snapshot(),
                    metadata={"payoff": payoff, "settlement_date": settlement_date, "schedule": schedule},
                )
        return None

    def _test_knock_in(self, date, product, env, spot, valuation_point) -> Optional[LifecycleEvent]:
        """The knock-in, only while schedule 1 is live (R2): never once its last observation has passed."""
        if self._due_at_previous_close(product, self._last_pre_record(product, env)):
            return None
        config = product.barrier_config
        ki_observation_type = getattr(config, "ki_observation_type", None)
        continuous = getattr(product, "has_ki_barrier", False) and (
            config.ki_continuous or getattr(ki_observation_type, "name", None) == "CONTINUOUS"
        )
        if continuous:
            barrier = config.ki_barrier[0] if isinstance(config.ki_barrier, list) else config.ki_barrier
            hits = [(None, float(barrier))] if self._barrier_hit(
                spot, float(barrier), product.is_reverse, is_ko=False) else []
            metadata = {"monitoring": "daily_close", "schedule": 1}
        else:
            hits = []
            for idx, rec in enumerate(self._scheduled_records(product, env, "ki")):
                if idx in self.lifecycle.observed_ki_indices or not self._record_is_due(date, valuation_point, rec):
                    continue
                self.lifecycle.observed_ki_indices.add(idx)
                if self._barrier_hit(spot, rec["barrier"], product.is_reverse, is_ko=False):
                    hits.append((idx, rec["barrier"]))
            metadata = {"schedule": 1}
        if not hits:
            return None
        index, barrier = hits[0]
        # the anchor is the record whose test fired, at its contractual fixing (accrual time spec §5.1); a
        # continuous knock-in has none (a REBASED contract requires discrete monitoring)
        anchor = None if index is None else knock_in_anchor_for(
            product.barrier_config.ki_observation_schedule.records[index], index)
        before = self._state_snapshot()
        if not self.lifecycle.mark_ki(pd.Timestamp(date).normalize().to_pydatetime(), anchor):
            return None
        return LifecycleEvent(
            event_type=LifecycleEventType.KNOCK_IN,
            date=date,
            spot=spot,
            observation_index=index,
            barrier=barrier,
            payoff=0.0,
            cashflow=0.0,
            terminates_position=False,
            state_before=before,
            state_after=self._state_snapshot(),
            metadata=metadata,
        )

    def _consume_schedule_two(self, env: PricingEnvironment) -> None:
        """At the knock-in close: schedule-2 records that fell due on an EARLIER close were due while schedule 2
        was not in force, so the knock-in consumes them (observed, never tested)."""
        contract = self.contract()
        for idx, rec in enumerate(self._schedule_records(contract, env, contract.post_barrier_config)):
            if self._due_at_previous_close(contract, rec):
                self.lifecycle.observed_post_ko_indices.add(idx)

    def settle_maturity_if_due(
        self, date: pd.Timestamp, product: Any, env: PricingEnvironment, spot: float
    ) -> Optional[LifecycleEvent]:
        """V0 at schedule 1's last observation (R2), or V1 at the knocked-in contract's end."""
        valuation_point = self._valuation_point(date, product)
        self.lifecycle.valuation_point = valuation_point
        if not self.lifecycle.alive:
            return None
        if self.lifecycle.knocked_in:
            timing = self._final_timing(env)
            if not self._timing_is_due(date, valuation_point, timing):
                return None
            contract = self.contract()
            payoff = float(contract.get_maturity_payoff_v1(
                spot, self._schedule_resolution_env(contract, env), contract_tenor=self.final_contract_tenor(env)
            ))
        else:
            last = self._last_pre_record(product, env)
            if not self._record_is_due(date, valuation_point, last):
                return None
            timing = self._pre_timing(product, env, last)
            payoff = float(product.get_maturity_payoff_v0(
                spot, self._schedule_resolution_env(product, env), contract_tenor=self.pre_contract_tenor(env)
            ))
        return self._book_maturity(date, spot, payoff, timing, valuation_point)

    def knock_in_anchor_on(self, dates: pd.DatetimeIndex, knock_in_day: int, env: PricingEnvironment):
        """The anchor of a knock-in observed on run-calendar position ``knock_in_day``, for a consumer that keys a
        REBASED path by its knock-in day (the simulated paths): the lowest-index discrete knock-in record due on that
        close, the record ``observe`` fires first. ``resolve_calendar_schedule`` refuses a REBASED contract whose
        records due on one close carry different levels, so a path knocked in on that close breached every one of
        them and this IS the record that fired (accrual time spec §5.1). ``None`` when no anchor is required."""
        product = self.product
        if not product.knock_in_anchor_required():
            return None
        records = self._scheduled_records(product, env, "ki")
        due = [
            index for index, rec in enumerate(records)
            if self._first_due(dates, lambda day, vp, r=rec: self._record_is_due(day, vp, r), product) == knock_in_day
        ]
        if not due:
            raise ValidationError(f"KO-reset snowball: no knock-in record is due on run day {knock_in_day}")
        index = due[0]
        return knock_in_anchor_for(product.barrier_config.ki_observation_schedule.records[index], index)

    def resolve_calendar_schedule(self, dates: pd.DatetimeIndex, env: PricingEnvironment) -> "KOResetCalendarSchedule":
        """Map this contract's observations onto ``dates`` (one resolution), by the records and due rules
        ``observe`` and ``settle_maturity_if_due`` use; a REBASED schedule 2 is placed per knock-in day."""
        return _resolve_calendar_schedule(self, dates, env)


def _resolve_calendar_schedule(tracker: KOResetLifecycleTracker, dates, env) -> KOResetCalendarSchedule:
    dates = pd.DatetimeIndex(dates)
    product = tracker.product
    base = AutocallableLifecycleTracker.resolve_calendar_schedule(tracker, dates, env)

    def records_on_calendar(owner, contract, config):
        recs = owner._schedule_records(contract, env, config)
        due = [owner._first_due(dates, lambda day, vp, r=r: owner._record_is_due(day, vp, r), contract) for r in recs]
        return (
            np.array(due, dtype=np.int64),
            np.array([float(r["barrier"]) if r["barrier"] is not None else np.nan for r in recs], dtype=float),
            np.array([float(r["payoff"]) for r in recs], dtype=float),
            np.array([owner._settlement_position(dates, r["settlement_date"], d) for r, d in zip(recs, due)],
                     dtype=np.int64),
            np.array([float(r["settlement_time"]) - float(r["time"]) for r in recs], dtype=float),
        )

    def placement(owner: KOResetLifecycleTracker) -> SchedulePlacement:
        contract = owner.contract()
        due, barrier, payoff, settle, delay = records_on_calendar(owner, contract, contract.post_barrier_config)
        timing = owner._final_timing(env)
        final_due = owner._first_due(dates, lambda day, vp: owner._timing_is_due(day, vp, timing), contract)
        return SchedulePlacement(
            ko_due_day=due, ko_barrier=barrier, ko_payoff=payoff, ko_settlement_day=settle, ko_settlement_delay=delay,
            final_due_day=int(final_due),
            final_settlement_day=int(owner._settlement_position(
                dates, owner._terminal_payment_date(contract, timing), final_due)),
            final_settlement_delay=float(timing.payment_time) - float(timing.determination_time),
            final_tenor=owner.final_contract_tenor(env),
        )

    rebased = product.post_ko_mode == PostKOScheduleMode.REBASED
    absolute = None
    place = None
    if rebased:
        _refuse_ambiguous_anchors(base, product)
        memo: Dict[int, SchedulePlacement] = {}

        def place(knock_in_day: int) -> SchedulePlacement:
            hit = memo.get(knock_in_day)
            if hit is None:
                scratch = KOResetLifecycleTracker(
                    product=product, quantity=tracker.quantity,
                    lifecycle=AutocallableLifecycleState(
                        knocked_in=True, ki_date=pd.Timestamp(dates[knock_in_day]).normalize().to_pydatetime(),
                        knock_in_anchor=tracker.knock_in_anchor_on(dates, knock_in_day, env)),
                    start_date=tracker.start_date, date_resolver=tracker._date_resolver,
                )
                hit = memo[knock_in_day] = placement(scratch)
            return hit
    else:
        absolute = placement(tracker)

    ko_due, ko_barrier, ko_payoff, ko_settle, ko_delay = records_on_calendar(tracker, product, product.barrier_config)
    last = tracker._last_pre_record(product, env)
    pre_due = tracker._first_due(dates, lambda day, vp: tracker._record_is_due(day, vp, last), product)
    pre_timing = tracker._pre_timing(product, env, last)
    pre_settle = tracker._settlement_position(dates, tracker._terminal_payment_date(product, pre_timing), pre_due)
    pre_delay = float(pre_timing.payment_time) - float(pre_timing.determination_time)

    # the latest terminal the calendar may have to hold: the knocked-in maturity of the last knock-in day
    if rebased:
        knock_in_days = [int(d) for d in base.ki_due_day if d >= 0 and (pre_due < 0 or d <= pre_due)]
        latest = place(max(knock_in_days)) if knock_in_days else None
    else:
        latest = absolute
    if latest is not None:
        terminal = (latest.final_due_day, latest.final_settlement_day, latest.final_settlement_delay)
    else:
        terminal = (int(pre_due), int(pre_settle), pre_delay)

    values = {f.name: getattr(base, f.name) for f in fields(CalendarSchedule)}
    values.update(
        ko_due_day=ko_due, ko_barrier=ko_barrier, ko_payoff=ko_payoff, ko_settlement_day=ko_settle,
        ko_settlement_delay=ko_delay, terminal_due_day=int(terminal[0]), terminal_settlement_day=int(terminal[1]),
        terminal_settlement_delay=float(terminal[2]),
    )
    return KOResetCalendarSchedule(
        **values, pre_due_day=int(pre_due), pre_settlement_day=int(pre_settle), pre_settlement_delay=pre_delay,
        pre_tenor=tracker.pre_contract_tenor(env), rebased=rebased, absolute=absolute, place=place,
    )


def _refuse_ambiguous_anchors(base: CalendarSchedule, product: Any) -> None:
    """A REBASED path is keyed by its knock-in day. When knock-in records due on one close carry different levels,
    which of them fired -- the anchor its accrual hangs from -- depends on the path's spot, not on the day."""
    if not product.knock_in_anchor_required():
        return
    levels: Dict[int, set] = {}
    for day, level in zip(base.ki_due_day, base.ki_barrier):
        if day >= 0:
            levels.setdefault(int(day), set()).add(float(level))
    clash = sorted(day for day, seen in levels.items() if len(seen) > 1)
    if clash:
        raise ValidationError(
            "KO-reset snowball (REBASED): knock-in records with different levels fall due on one close of the run "
            f"calendar (positions {clash}); the record that fires, and so the anchor schedule 2 accrues from, would "
            "depend on the path's spot. Simulated paths key a knocked-in path by its knock-in day only."
        )


def _last_observation_date(schedule) -> Optional[Any]:
    """A dated schedule's last observation date (its contract end); ``None`` for a numeric one."""
    dates = [] if schedule is None else [r.observation_date for r in schedule.records if r.observation_date is not None]
    return max(dates) if dates else None


def tracker_for(product: Any, **kwargs) -> AutocallableLifecycleTracker:
    """The tracker a product's rules need: KO-reset or the snowball/Phoenix one."""
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption

    cls = KOResetLifecycleTracker if isinstance(product, KnockOutResetSnowballOption) else AutocallableLifecycleTracker
    return cls(product=product, **kwargs)
