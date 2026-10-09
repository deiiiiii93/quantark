"""Shared machinery of the FX target-redemption products (TARN and TARF).

Spec: docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md, sections 3-6. A contract carries its realized
state (``state``: the amount banked towards the target and the determined cash still to pay). In the NUMERIC form
``fixing_times`` / ``pay_times`` are years from valuation and list ONLY the fixings not yet banked; a time at or
before valuation is an unbanked past fixing. In the DATED form ``fixing_dates`` / ``pay_dates`` list the whole
contractual schedule and ``state.as_of`` says through which date the fixings are banked; times resolve against the
environment at pricing. Only the two concrete products are instantiated.
"""
from __future__ import annotations

import math
import warnings
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
from typing import List, Optional, Tuple

from quantark.util.calendar import calculate_year_fraction
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import fixing_label, warn_filled_fixings
from quantark.util.numerical import is_zero

from ..base_fx_product import BaseFxProduct
from .target_redemption_state import (
    PendingPayment,
    ResolvedTargetRedemption,
    TargetRedemptionEvent,
    TargetRedemptionEventType,
    TargetRedemptionState,
    TerminalBookingWarning,
    is_target_hit,
    pending_present_value,
    target_hit_tol,
)


def is_passed_time(t: float) -> bool:
    """A numeric fixing or pay time at or before valuation; ``is_zero`` absorbs the ULPs daily ageing leaves."""
    return t <= 0.0 or is_zero(t)


def signed_year_fraction(date: datetime, fx_env) -> float:
    """Signed year fraction from ``fx_env.valuation_date`` to ``date`` in the ENVIRONMENT's convention: 0.0 on the
    valuation date, negative before it (``FxRangeAccrualObservationRecord.resolve_time``'s rule)."""
    val = fx_env.valuation_date
    if date == val:
        return 0.0
    if date > val:
        return calculate_year_fraction(val, date, fx_env.day_count_convention, fx_env.bus_days_in_year,
                                       calendar=fx_env.calendar)
    return -calculate_year_fraction(date, val, fx_env.day_count_convention, fx_env.bus_days_in_year,
                                    calendar=fx_env.calendar)


def event_label(index: int, *, time: Optional[float] = None, date: Optional[datetime] = None) -> str:
    """W5 ``fixing_label`` text (``#i t=<time>`` numeric, ``#i <date>`` dated) for a TARN/TARF fixing."""
    return fixing_label(index, SimpleNamespace(observation_time=time, observation_date=date))


class TargetRedemptionProduct(BaseFxProduct):
    """Base of ``FxTargetRedemptionNote`` and ``FxTargetRedemptionForward``: schedule forms, realized state,
    validation, terminal bookings, ageing and the resolved view."""

    #: TARN: per-period accrual factors aligned with the listed fixings.
    _has_accrual: bool = False
    #: TARN: determined cash is a coupon or the principal, so a pending amount must be > 0.
    _pending_must_be_positive: bool = False

    fixing_times: Optional[List[float]]
    pay_times: Optional[List[float]]
    fixing_dates: Optional[List[datetime]]
    pay_dates: Optional[List[datetime]]
    lifecycle_events: Tuple[TargetRedemptionEvent, ...]

    # -- subclass contract --------------------------------------------------------------------------------------
    def _validate_terms(self) -> None:
        raise NotImplementedError

    def _fixing_outcome(self, factor: Optional[float], spot: float, accumulated: float) -> Tuple[float, float]:
        """``(amount banked towards the target, cash determined)`` for one fixing at ``spot``."""
        raise NotImplementedError

    def _principal_amount(self) -> float:
        """Cash due when the contract ends (the TARN principal); 0.0 when none."""
        return 0.0

    # -- construction -------------------------------------------------------------------------------------------
    @staticmethod
    def _require_state(state) -> TargetRedemptionState:
        if state is None:
            raise ValidationError(
                "state is required: a contract booked at inception passes TargetRedemptionState(0.0); a mid-life "
                "booking states what it has banked and what is still to pay"
            )
        if not isinstance(state, TargetRedemptionState):
            raise ValidationError(f"state must be a TargetRedemptionState, got {type(state).__name__}")
        return state

    @staticmethod
    def _schedule_form(fixing_times, fixing_dates) -> bool:
        """True for the dated form; exactly one schedule is required."""
        if fixing_times is not None and fixing_dates is not None:
            raise ValidationError("give fixing_times (numeric form) or fixing_dates (dated form), not both")
        if fixing_times is None and fixing_dates is None:
            raise ValidationError("a schedule is required: fixing_times (numeric form) or fixing_dates (dated form)")
        return fixing_dates is not None

    def _init_dated(self, fixing_dates, pay_dates, currency_pair, state: TargetRedemptionState) -> None:
        BaseFxProduct.__init__(self, currency_pair=currency_pair)
        self.fixing_dates = list(fixing_dates)
        self.pay_dates = list(pay_dates) if pay_dates is not None else list(self.fixing_dates)
        self.fixing_times = None
        self.pay_times = None
        self.state = state
        self.lifecycle_events = ()

    def _refresh_dated_horizon(self) -> None:
        """``expiry_date`` = last fixing date, ``delivery_date`` = last pay date; ``maturity`` stays None."""
        self.expiry_date = self.fixing_dates[-1]
        self.delivery_date = max(self.pay_dates)

    def _init_numeric(self, fixings: List[float], pay_times, currency_pair, state: TargetRedemptionState) -> None:
        BaseFxProduct.__init__(self, currency_pair=currency_pair)
        self.fixing_times = list(fixings)
        self.pay_times = [float(t) for t in pay_times] if pay_times is not None else list(fixings)
        self.fixing_dates = None
        self.pay_dates = None
        self.state = state
        self.lifecycle_events = ()

    def _refresh_numeric_horizon(self) -> None:
        """``maturity`` = last listed fixing, else the last pending pay time, else 0.0; ``delivery`` alike."""
        if self.fixing_times:
            self.maturity = self.fixing_times[-1]
            self.delivery = max(self.pay_times)
        else:
            pending = [p.pay_time for p in self.state.pending]
            self.maturity = self.delivery = max(pending) if pending else 0.0

    def _warn_if_terminal_booking(self) -> None:
        if self.is_terminal:
            why = "redeemed" if self.is_redeemed else "every fixing banked"
            warnings.warn(
                TerminalBookingWarning(
                    f"{type(self).__name__} booked terminal ({why}): it is valued at the PV of its "
                    f"{len(self.state.pending)} pending payment(s), 0.0 when none"
                ),
                stacklevel=3,
            )

    # -- derived flags ------------------------------------------------------------------------------------------
    @property
    def is_numeric(self) -> bool:
        return self.fixing_dates is None

    @property
    def hit_tol(self) -> float:
        """Target-hit tolerance shared with the engines (``target_hit_tol``)."""
        return target_hit_tol(self.target)

    @property
    def is_unbounded_target(self) -> bool:
        return math.isinf(self.target)

    @property
    def is_redeemed(self) -> bool:
        return is_target_hit(self.state.accumulated, self.target)

    @property
    def is_terminal(self) -> bool:
        """Redeemed, or no fixing left: numeric - an empty list; dated - ``as_of`` on or after the last fixing."""
        if self.is_numeric:
            return not self.fixing_times or self.is_redeemed
        as_of = self.state.as_of
        return self.is_redeemed or (as_of is not None and as_of >= self.fixing_dates[-1])

    # -- validation ---------------------------------------------------------------------------------------------
    def validate(self) -> None:
        self._validate_terms()
        if self.is_numeric:
            self._validate_numeric_schedule()
        else:
            self._validate_dated_schedule()
        self._validate_state()

    def _validate_dated_accrual(self) -> None:
        """TARN hook: the dated period source."""

    def _validate_dated_schedule(self) -> None:
        dates, pays = self.fixing_dates, self.pay_dates
        if not dates:
            raise ValidationError("fixing_dates must be non-empty: the dated form lists the whole schedule")
        if any(not isinstance(d, datetime) for d in list(dates) + list(pays)):
            raise ValidationError("fixing_dates and pay_dates must be datetimes")
        if dates != sorted(dates):
            raise ValidationError("fixing_dates must be ascending")
        if len(set(dates)) != len(dates):
            raise ValidationError("fixing_dates must be unique")
        if len(pays) != len(dates):
            raise ValidationError("pay_dates must align 1:1 with fixing_dates")
        for f, p in zip(dates, pays):
            if p < f:
                raise ValidationError(f"pay date {p.date()} precedes its fixing {f.date()}")
        if pays != sorted(pays):
            raise ValidationError("pay_dates must be non-decreasing")
        self._validate_dated_accrual()

    def _validate_numeric_schedule(self) -> None:
        fixings, pays = self.fixing_times, self.pay_times
        if list(fixings) != sorted(fixings):
            raise ValidationError("fixing_times must be ascending")
        if len(set(fixings)) != len(fixings):
            raise ValidationError("fixing_times must be unique")
        if any(not math.isfinite(t) for t in fixings):
            raise ValidationError("fixing_times must be finite")
        if len(pays) != len(fixings):
            raise ValidationError("pay_times must align 1:1 with fixing_times")
        for tf, tp in zip(fixings, pays):
            if not math.isfinite(tp):
                raise ValidationError(f"pay time {tp} must be finite")
            if tp < tf:
                raise ValidationError(f"pay time {tp} precedes its fixing {tf}")
        if list(pays) != sorted(pays):
            raise ValidationError("pay_times must be non-decreasing")
        if self._has_accrual:
            factors = self.accrual_factors
            if len(factors) != len(fixings):
                raise ValidationError("accrual_factors must align 1:1 with fixing_times")
            if any((not math.isfinite(a)) or a <= 0.0 for a in factors):
                raise ValidationError("accrual_factors must be positive and finite")

    def _validate_state(self) -> None:
        st = self.state
        if st.accumulated > self.target + self.hit_tol:
            raise ValidationError(f"state.accumulated {st.accumulated} exceeds the target {self.target}")
        if self._pending_must_be_positive and any(p.amount <= 0.0 for p in st.pending):
            raise ValidationError(
                "pending amounts must be positive: a TARN's determined cash is a coupon or its principal"
            )
        if self.is_numeric:
            if st.as_of is not None:
                raise ValidationError(
                    "state.as_of belongs to the dated form: a numeric state lists only unbanked fixings")
            if any(p.pay_date is not None for p in st.pending):
                raise ValidationError("a numeric contract's pending payments take pay_time, not pay_date")
            if self.is_redeemed and self.fixing_times:
                raise ValidationError(
                    "a redeemed state cannot list fixings: every fixing after the target hit is dead")
            return
        if any(p.pay_time is not None for p in st.pending):
            raise ValidationError("a dated contract's pending payments take pay_date, not pay_time")
        if (st.accumulated > 0.0 or st.pending) and (st.as_of is None or st.as_of < self.fixing_dates[0]):
            raise ValidationError(
                "accumulated or pending cash needs state.as_of on or after the first fixing date")

    # -- dated helpers ------------------------------------------------------------------------------------------
    def _dated_factor(self, index: int, fx_env) -> Optional[float]:
        """TARN hook: the accrual factor of scheduled period ``index``; None for the TARF."""
        return None

    def _is_unbanked(self, index: int) -> bool:
        """Dated: fixing ``index`` is banked iff it is dated on or before ``state.as_of``."""
        as_of = self.state.as_of
        return as_of is None or self.fixing_dates[index] > as_of

    def _check_as_of(self, fx_env) -> None:
        as_of = self.state.as_of
        if as_of is not None and as_of > fx_env.valuation_date:
            raise ValidationError(
                f"state.as_of {as_of} is after the valuation date {fx_env.valuation_date}: the state claims "
                "fixings that have not happened")

    def get_maturity(self, fx_env=None) -> float:
        """Dated: 0.0 once valuation is past the last fixing date (an aged contract still answers FX consumers)."""
        if self.expiry_date is not None and fx_env is not None and fx_env.valuation_date > self.expiry_date:
            return 0.0
        return super().get_maturity(fx_env)

    def get_delivery(self, fx_env=None) -> float:
        """Dated: 0.0 once valuation is past the last pay date (Ruling R8: ``calculate_greeks`` reads it)."""
        if self.delivery_date is not None and fx_env is not None and fx_env.valuation_date > self.delivery_date:
            return 0.0
        return super().get_delivery(fx_env)

    # -- ageing (spec section 6) --------------------------------------------------------------------------------
    def time_shift(self, time_bump: float) -> None:
        """Numeric form: move every fixing, pay and pending time by ``time_bump``. Pending cash now at or before
        valuation was paid within the bump and leaves as a PAYMENT event; fixings stepped over stay unbanked until
        :meth:`record_passed_fixings`. The dated form ages by its environment's date, so this is a no-op for it."""
        if not self.is_numeric:
            return
        self.fixing_times = [t - time_bump for t in self.fixing_times]
        self.pay_times = [t - time_bump for t in self.pay_times]
        kept, events = [], []
        for p in self.state.pending:
            t = p.pay_time - time_bump
            if is_passed_time(t):
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, p.amount, time=t))
            else:
                kept.append(PendingPayment(p.amount, pay_time=t))
        self.state = TargetRedemptionState(self.state.accumulated, pending=tuple(kept))
        self.lifecycle_events = self.lifecycle_events + tuple(events)
        self._refresh_numeric_horizon()
        self.validate()

    def record_passed_fixings(self, fx_env) -> None:
        """Bank every unbanked fixing at or before valuation, in order, at the QUOTED ``fx_env.spot`` (spec Q3),
        stopping once the contract is terminal; append the events; re-validate. Dated: ``state.as_of`` becomes the
        valuation date and pending cash dated on or before it leaves as a PAYMENT event."""
        if self.is_numeric:
            self._record_numeric(fx_env)
        else:
            self._record_dated(fx_env)
        self.validate()

    def _record_numeric(self, fx_env) -> None:
        spot = fx_env.spot
        acc = self.state.accumulated
        pending = list(self.state.pending)
        events: List[TargetRedemptionEvent] = []
        fixings, pays = self.fixing_times, self.pay_times
        factors = self.accrual_factors if self._has_accrual else None
        banked = 0
        while banked < len(fixings) and is_passed_time(fixings[banked]):
            t, tp = fixings[banked], pays[banked]
            added, cash = self._fixing_outcome(factors[banked] if factors is not None else None, spot, acc)
            acc = acc + added
            hit = is_target_hit(acc, self.target)
            ends = hit or banked == len(fixings) - 1
            label = event_label(banked, time=t)
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.FIXING, cash, fixing=spot, time=t,
                                                redeemed=ends, label=label))
            self._settle_numeric(cash, tp, label, pending, events)
            if ends:
                self._settle_numeric(self._principal_amount(), tp, label, pending, events)
            banked += 1
            if hit:
                banked = len(fixings)  # every later fixing is dead
                break
        self.fixing_times = fixings[banked:]
        self.pay_times = pays[banked:]
        if factors is not None:
            self.accrual_factors = factors[banked:]
        self.state = TargetRedemptionState(acc, pending=tuple(sorted(pending, key=lambda p: p.pay_time)))
        self.lifecycle_events = self.lifecycle_events + tuple(events)
        self._refresh_numeric_horizon()

    @staticmethod
    def _settle_numeric(amount: float, pay_time: float, label: str, pending: list, events: list) -> None:
        """Determined cash: paid now (PAYMENT, the fixing's label) when its pay time has passed, else pending."""
        if is_zero(amount):
            return
        if is_passed_time(pay_time):
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, amount, time=pay_time,
                                                label=label))
        else:
            pending.append(PendingPayment(amount, pay_time=pay_time))

    def _record_dated(self, fx_env) -> None:
        self._check_as_of(fx_env)
        val, spot, st = fx_env.valuation_date, fx_env.spot, self.state
        events: List[TargetRedemptionEvent] = []
        pending: list = []
        for p in st.pending:  # cash due by the valuation date was paid
            if p.pay_date <= val:
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, p.amount,
                                                    time=signed_year_fraction(p.pay_date, fx_env)))
            else:
                pending.append(p)
        acc = st.accumulated
        if not self.is_terminal:
            last = len(self.fixing_dates) - 1
            for i, fd in enumerate(self.fixing_dates):
                if not self._is_unbanked(i):
                    continue
                if fd > val:
                    break
                t = signed_year_fraction(fd, fx_env)
                added, cash = self._fixing_outcome(self._dated_factor(i, fx_env), spot, acc)
                acc = acc + added
                hit = is_target_hit(acc, self.target)
                ends = hit or i == last
                label = event_label(i, date=fd)
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.FIXING, cash, fixing=spot, time=t,
                                                    redeemed=ends, label=label))
                self._settle_dated(cash, self.pay_dates[i], label, fx_env, pending, events)
                if ends:
                    self._settle_dated(self._principal_amount(), self.pay_dates[i], label, fx_env, pending, events)
                    break  # every later fixing is dead (or there is none)
        self.state = TargetRedemptionState(acc, pending=tuple(sorted(pending, key=lambda p: p.pay_date)), as_of=val)
        self.lifecycle_events = self.lifecycle_events + tuple(events)

    @staticmethod
    def _settle_dated(amount: float, pay_date: datetime, label: str, fx_env, pending: list, events: list) -> None:
        """Dated twin of ``_settle_numeric``: paid now when ``pay_date`` is on or before valuation, else pending."""
        if is_zero(amount):
            return
        if pay_date <= fx_env.valuation_date:
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, amount,
                                                time=signed_year_fraction(pay_date, fx_env), label=label))
        else:
            pending.append(PendingPayment(amount, pay_date=pay_date))

    # -- resolved view (spec section 5.1) ----------------------------------------------------------------------
    def resolve(self, fx_env, *, warn_missing_fixings: bool = True) -> ResolvedTargetRedemption:
        """The engines' only input. Unbanked fixings at or before valuation are banked on a COPY at the quoted spot
        (``record_passed_fixings``) with one ``MissingFixingWarning`` naming them (W5); the product is unchanged."""
        if self._has_unbanked_passed(fx_env):
            banked = deepcopy(self)
            start = len(banked.lifecycle_events)
            banked.record_passed_fixings(fx_env)
            if warn_missing_fixings:
                labels = [e.label for e in banked.lifecycle_events[start:]
                          if e.event_type is TargetRedemptionEventType.FIXING]
                warn_filled_fixings(type(self).__name__, "a banked outcome", labels)
            return banked._view(fx_env)
        return self._view(fx_env)

    def _has_unbanked_passed(self, fx_env) -> bool:
        if self.is_numeric:
            return bool(self.fixing_times) and is_passed_time(self.fixing_times[0])
        self._check_as_of(fx_env)
        if self.is_terminal:
            return False
        val = fx_env.valuation_date
        return any(self._is_unbanked(i) and d <= val for i, d in enumerate(self.fixing_dates))

    def _view(self, fx_env) -> ResolvedTargetRedemption:
        if not self.is_numeric:
            return self._dated_view(fx_env)
        st = self.state
        return ResolvedTargetRedemption(
            fixing_times=tuple(self.fixing_times),
            pay_times=tuple(self.pay_times),
            accrual_factors=tuple(self.accrual_factors) if self._has_accrual else None,
            accumulated=st.accumulated,
            pending=tuple((p.pay_time, p.amount) for p in st.pending),
            redeemed=self.is_redeemed,
            terminal=self.is_terminal,
        )

    def _dated_view(self, fx_env) -> ResolvedTargetRedemption:
        """Unbanked fixings (all after valuation once ``resolve`` has banked the passed ones), their pay times and
        factors resolved against ``fx_env``; pending cash dated after valuation. Dead fixings of a redeemed
        contract are dropped."""
        val, st = fx_env.valuation_date, self.state
        fix_t: List[float] = []
        pay_t: List[float] = []
        factors: List[float] = []
        if not self.is_terminal:
            for i, d in enumerate(self.fixing_dates):
                if not self._is_unbanked(i):
                    continue
                t = signed_year_fraction(d, fx_env)
                if is_passed_time(t):  # Ruling R7: after valuation by date, zero by the environment's day count
                    raise ValidationError(
                        f"fixing date {d.date()} is after the valuation date {val} but resolves to a zero year "
                        f"fraction under the environment's {fx_env.day_count_convention.name} day count")
                fix_t.append(t)
                pay_t.append(signed_year_fraction(self.pay_dates[i], fx_env))
                if self._has_accrual:
                    factors.append(self._dated_factor(i, fx_env))
        pending: List[Tuple[float, float]] = []
        for p in st.pending:
            if p.pay_date <= val:
                continue  # paid: booked, not pending
            t = signed_year_fraction(p.pay_date, fx_env)
            if is_passed_time(t):  # Ruling R7 applied to cash: unpaid by date, paid by the numeric rule
                raise ValidationError(
                    f"pending pay date {p.pay_date.date()} is after the valuation date {val} but resolves to a zero "
                    f"year fraction under the environment's {fx_env.day_count_convention.name} day count")
            pending.append((t, p.amount))
        return ResolvedTargetRedemption(
            fixing_times=tuple(fix_t),
            pay_times=tuple(pay_t),
            accrual_factors=tuple(factors) if self._has_accrual else None,
            accumulated=st.accumulated,
            pending=tuple(pending),
            redeemed=self.is_redeemed,
            terminal=self.is_terminal,
        )

    def pending_pv(self, fx_env) -> float:
        """PV of the pending cash still due after valuation (no warning: callers have banked the passed fixings)."""
        return pending_present_value(self.resolve(fx_env, warn_missing_fixings=False).pending, fx_env)
