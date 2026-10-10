"""Shared machinery of the FX target-redemption products (TARN and TARF).

Spec: docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md, sections 3-6. A contract carries its realized
state (``state``: the amount banked towards the target and the determined cash still to pay). In the NUMERIC form
``fixing_times`` / ``pay_times`` are years from valuation and list ONLY the fixings not yet banked; a time at or
before valuation is an unbanked past fixing. In the DATED form ``fixing_dates`` / ``pay_dates`` list the whole
contractual schedule; the state counts the banked fixings (``state.fixings_banked``, per fixing in schedule order, user
ruling OQ-3 2026-10-10; a booking stating only ``as_of`` banks the fixings dated on or before it) and ``state.as_of``
is the date it was brought up to date, never after the valuation date. Times resolve against the environment at
pricing; passed vs ahead is decided by TIME, so a fixing or pay date at zero time from valuation (a Saturday under
``BUSINESS_DAYS``, valued on the Friday) is live like its numeric twin (user decision OQ-2 b, 2026-10-10). Only the
two concrete products are instantiated.

A passed fixing's history is its fixing RATE. An unbanked passed fixing without one is banked at the quoted spot of
whatever environment prices it (spec Q3, one ``MissingFixingWarning``); the W5 fill protocol
(``missing_fixing_fills`` / ``with_fixings_recorded`` / ``with_missing_fixings_filled``) records that rate once, at
the BASE environment, on a copy, so bumped and shocked repricings bank the fixing at the base spot (user decision
2026-10-09: a filled past fixing is history and does not move).
"""
from __future__ import annotations

import math
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Tuple

from quantark.util.calendar import calculate_year_fraction
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import fixing_label, is_history, snap_time, warn_filled_fixings
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
        self._recorded_rates: Dict[int, float] = {}

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
        self._recorded_rates: Dict[int, float] = {}

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
        """Redeemed, or no fixing left: numeric - an empty list; dated - every scheduled fixing banked."""
        if self.is_numeric:
            return not self.fixing_times or self.is_redeemed
        return self.is_redeemed or self._banked_count() >= len(self.fixing_dates)

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
            if st.fixings_banked is not None:
                raise ValidationError(
                    "state.fixings_banked belongs to the dated form: a numeric state lists only unbanked fixings")
            if any(p.pay_date is not None for p in st.pending):
                raise ValidationError("a numeric contract's pending payments take pay_time, not pay_date")
            if self.is_redeemed and self.fixing_times:
                raise ValidationError(
                    "a redeemed state cannot list fixings: every fixing after the target hit is dead")
            return
        if any(p.pay_time is not None for p in st.pending):
            raise ValidationError("a dated contract's pending payments take pay_date, not pay_time")
        if st.fixings_banked is not None and st.fixings_banked > len(self.fixing_dates):
            raise ValidationError(
                f"state.fixings_banked {st.fixings_banked} exceeds the {len(self.fixing_dates)} scheduled fixings")
        # cash implies a banked fixing (for a booking stating only as_of: as_of on or after the first fixing date)
        if (st.accumulated > 0.0 or st.pending) and (st.as_of is None or self._banked_count() == 0):
            raise ValidationError(
                "accumulated or pending cash needs state.as_of and at least one banked fixing")

    # -- dated helpers ------------------------------------------------------------------------------------------
    def _dated_factor(self, index: int, fx_env) -> Optional[float]:
        """TARN hook: the accrual factor of scheduled period ``index``; None for the TARF."""
        return None

    def _banked_count(self) -> int:
        """Dated: how many leading scheduled fixings are banked (OQ-3 ruling, per fixing): ``state.fixings_banked``,
        else - a booking that states only ``as_of`` - the fixings dated on or before ``as_of``, else 0."""
        st = self.state
        if st.fixings_banked is not None:
            return st.fixings_banked
        if st.as_of is None:
            return 0
        return sum(1 for d in self.fixing_dates if d <= st.as_of)

    @staticmethod
    def _state_date(fx_env) -> datetime:
        """The ``as_of`` banking writes (Ruling R24): the valuation date, moved back over the days at zero time from it
        (valued on a Saturday under BUSINESS_DAYS, the Friday: the same instant), so it never passes the valuation date
        and cash due at that instant never precedes it. Under CALENDAR_DAYS / ACT conventions: the valuation date."""
        val = fx_env.valuation_date
        d = val
        for _ in range(31):  # bounded: no zero-time stretch (weekend + holidays) is a month long
            prev = d - timedelta(days=1)
            if not is_zero(calculate_year_fraction(prev, val, fx_env.day_count_convention, fx_env.bus_days_in_year,
                                                   calendar=fx_env.calendar)):
                break
            d = prev
        return d

    def _is_unbanked(self, index: int) -> bool:
        """Dated: fixing ``index`` is banked iff it is one of the first ``_banked_count()`` (identity, not date)."""
        return index >= self._banked_count()

    def _check_as_of(self, fx_env) -> None:
        """Dated: the state may not claim the future. ``as_of`` after the valuation date is refused, and so is a banked
        count whose last banked fixing lies strictly after the valuation instant, decided by TIME (Ruling R16): a
        fixing at zero time from valuation (a Saturday under BUSINESS_DAYS, valued on the Friday) may be banked."""
        as_of = self.state.as_of
        if as_of is not None and as_of > fx_env.valuation_date:
            raise ValidationError(
                f"state.as_of {as_of} is after the valuation date {fx_env.valuation_date}: the state claims "
                "fixings that have not happened")
        k = self._banked_count()
        if k > 0 and not is_passed_time(signed_year_fraction(self.fixing_dates[k - 1], fx_env)):
            raise ValidationError(
                f"state.fixings_banked {k} claims fixing {self.fixing_dates[k - 1].date()}, which is after the "
                f"valuation instant {fx_env.valuation_date} under the environment's "
                f"{fx_env.day_count_convention.name} day count: the state claims fixings that have not happened")

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
        """Numeric form: move every fixing, pay and pending time by ``time_bump``. Pending cash whose pay time is now
        strictly past was paid within the bump and leaves as a PAYMENT event; cash due today (snapped to exactly 0.0)
        stays pending, in the value (user decision OQ-1 b, 2026-10-10). Fixings stepped over stay unbanked until
        :meth:`record_passed_fixings`. The dated form ages by its environment's date, so this is a no-op for it."""
        if not self.is_numeric:
            return
        self.fixing_times = [t - time_bump for t in self.fixing_times]
        self.pay_times = [t - time_bump for t in self.pay_times]
        kept, events = [], []
        for p in self.state.pending:
            t = p.pay_time - time_bump
            if is_history(t):  # strictly past: paid within the bump (OQ-1 b: cash due today stays pending)
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, p.amount, time=t))
            else:
                kept.append(PendingPayment(p.amount, pay_time=snap_time(t)))
        self.state = TargetRedemptionState(self.state.accumulated, pending=tuple(kept))
        self.lifecycle_events = self.lifecycle_events + tuple(events)
        self._refresh_numeric_horizon()
        self.validate()

    def record_passed_fixings(self, fx_env) -> None:
        """Bank every unbanked fixing at or before valuation, in order, at its recorded rate (``with_fixings_recorded``)
        or else the QUOTED ``fx_env.spot`` (spec Q3), stopping once the contract is terminal; append the events;
        re-validate. Dated: passed is decided by TIME (a fixing at zero time from valuation is banked, OQ-2 b);
        ``state.fixings_banked`` counts the banked fixings and ``state.as_of`` becomes the valuation date, or the first
        date of its zero-time stretch (Ruling R24: a Saturday under BUSINESS_DAYS writes the Friday); pending cash
        whose pay time is strictly past leaves as a PAYMENT event; cash due at the valuation instant stays pending
        (OQ-1 b)."""
        self._bank_passed(fx_env)
        self.validate()

    def _bank_passed(self, fx_env) -> List[Tuple[int, str]]:
        """``record_passed_fixings`` without the re-validation; returns ``(index, label)`` of every fixing banked at
        ``fx_env.spot`` because it had no recorded rate (the fills; index into the schedule as it was listed)."""
        if self.is_numeric:
            return self._record_numeric(fx_env)
        return self._record_dated(fx_env)

    def _rate(self, index: int, fx_env, filled: List[Tuple[int, str]], label: str) -> float:
        """Fixing ``index``'s recorded rate, else the quoted spot (noted in ``filled``)."""
        rate = self._recorded_rates.get(index)
        if rate is not None:
            return rate
        filled.append((index, label))
        return fx_env.spot

    def _record_numeric(self, fx_env) -> List[Tuple[int, str]]:
        filled: List[Tuple[int, str]] = []
        acc = self.state.accumulated
        pending = list(self.state.pending)
        events: List[TargetRedemptionEvent] = []
        fixings, pays = self.fixing_times, self.pay_times
        factors = self.accrual_factors if self._has_accrual else None
        banked = 0
        while banked < len(fixings) and is_passed_time(fixings[banked]):
            t, tp = fixings[banked], pays[banked]
            label = event_label(banked, time=t)
            spot = self._rate(banked, fx_env, filled, label)
            added, cash = self._fixing_outcome(factors[banked] if factors is not None else None, spot, acc)
            acc = acc + added
            hit = is_target_hit(acc, self.target)
            ends = hit or banked == len(fixings) - 1
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
        # the listed schedule lost its first ``banked`` fixings: recorded rates follow their fixings
        self._recorded_rates = {i - banked: r for i, r in self._recorded_rates.items() if i >= banked}
        self.state = TargetRedemptionState(acc, pending=tuple(sorted(pending, key=lambda p: p.pay_time)))
        self.lifecycle_events = self.lifecycle_events + tuple(events)
        self._refresh_numeric_horizon()
        return filled

    @staticmethod
    def _settle_numeric(amount: float, pay_time: float, label: str, pending: list, events: list) -> None:
        """Determined cash: paid now (PAYMENT, the fixing's label) when its pay time is strictly past, else pending
        (cash due today included, at exactly 0.0)."""
        if is_zero(amount):
            return
        if is_history(pay_time):
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, amount, time=pay_time,
                                                label=label))
        else:
            pending.append(PendingPayment(amount, pay_time=snap_time(pay_time)))

    def _record_dated(self, fx_env) -> List[Tuple[int, str]]:
        self._check_as_of(fx_env)
        filled: List[Tuple[int, str]] = []
        st = self.state
        events: List[TargetRedemptionEvent] = []
        pending: list = []
        for p in st.pending:  # cash whose pay time is strictly past was paid; cash due today stays pending
            t_pay = signed_year_fraction(p.pay_date, fx_env)
            if is_history(t_pay):
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, p.amount, time=t_pay))
            else:
                pending.append(p)
        acc = st.accumulated
        banked = self._banked_count()
        if not self.is_terminal:
            last = len(self.fixing_dates) - 1
            for i in range(banked, len(self.fixing_dates)):
                fd = self.fixing_dates[i]
                t = signed_year_fraction(fd, fx_env)
                if not is_passed_time(t):  # ahead by TIME (OQ-2 b): a Saturday at zero business time is passed
                    break
                label = event_label(i, date=fd)
                spot = self._rate(i, fx_env, filled, label)
                added, cash = self._fixing_outcome(self._dated_factor(i, fx_env), spot, acc)
                acc = acc + added
                hit = is_target_hit(acc, self.target)
                ends = hit or i == last
                events.append(TargetRedemptionEvent(TargetRedemptionEventType.FIXING, cash, fixing=spot, time=t,
                                                    redeemed=ends, label=label))
                self._settle_dated(cash, self.pay_dates[i], label, fx_env, pending, events)
                banked = i + 1  # identity: fixings 0 .. i are banked, whatever their dates (OQ-3 ruling)
                if ends:
                    self._settle_dated(self._principal_amount(), self.pay_dates[i], label, fx_env, pending, events)
                    break  # every later fixing is dead (or there is none)
        self.state = TargetRedemptionState(acc, pending=tuple(sorted(pending, key=lambda p: p.pay_date)),
                                           as_of=self._state_date(fx_env), fixings_banked=banked)
        self.lifecycle_events = self.lifecycle_events + tuple(events)
        # a banked or dead fixing's recorded rate is spent
        self._recorded_rates = {i: r for i, r in self._recorded_rates.items()
                                if not self.is_terminal and self._is_unbanked(i)}
        return filled

    @staticmethod
    def _settle_dated(amount: float, pay_date: datetime, label: str, fx_env, pending: list, events: list) -> None:
        """Dated twin of ``_settle_numeric``: paid now when ``pay_date``'s time is strictly past, else pending."""
        if is_zero(amount):
            return
        t = signed_year_fraction(pay_date, fx_env)
        if is_history(t):
            events.append(TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, amount, time=t, label=label))
        else:
            pending.append(PendingPayment(amount, pay_date=pay_date))

    # -- resolved view (spec section 5.1) ----------------------------------------------------------------------
    def resolve(self, fx_env, *, warn_missing_fixings: bool = True) -> ResolvedTargetRedemption:
        """The engines' only input. Unbanked fixings at or before valuation are banked on a COPY
        (``record_passed_fixings``): at their recorded rate, else at the quoted spot with one ``MissingFixingWarning``
        naming those (W5); the product is unchanged."""
        if self._has_unbanked_passed(fx_env):
            banked = deepcopy(self)
            filled = banked._bank_passed(fx_env)
            banked.validate()
            if warn_missing_fixings:
                warn_filled_fixings(type(self).__name__, "a banked outcome", [label for _, label in filled])
            return banked._view(fx_env)
        return self._view(fx_env)

    # -- W5 fill protocol (user decision 2026-10-09: a filled past fixing does not move) ------------------------
    def missing_fixing_fills(self, fx_env) -> Dict[int, float]:
        """``{index: fx_env.spot}`` for every unbanked passed fixing that banking would fill at the quoted spot (no
        recorded rate; fixings dead after a redemption inside the fill are not listed). ``index`` is the fixing's
        position in ``fixing_times`` (numeric: the unbanked fixings listed now) or ``fixing_dates`` (dated). No
        warning; the product is unchanged."""
        return {index: fx_env.spot for index, _ in self._passed_fills(fx_env)}

    def _passed_fills(self, fx_env) -> List[Tuple[int, str]]:
        """``(index, label)`` of the fixings banking at ``fx_env`` would fill at its spot, found on a probe copy."""
        if not self._has_unbanked_passed(fx_env):
            return []
        return deepcopy(self)._bank_passed(fx_env)

    def with_fixings_recorded(self, fills: Mapping[int, float]) -> "TargetRedemptionProduct":
        """A COPY with each listed unbanked fixing's RATE recorded from ``fills`` (``{index: rate}``, indexed as in
        ``missing_fixing_fills``); banking (``record_passed_fixings``, ``resolve``) then uses that rate, so the
        outcome follows the contract's state at banking (a recorded fill can redeem it). A fixing that already has a
        recorded rate keeps it; a banked or out-of-range index is skipped. Returns ``self`` when nothing changes; the
        product is never changed. A rate recorded for a fixing still after valuation is refused at ``resolve``."""
        rates = dict(self._recorded_rates)
        for index, rate in fills.items():
            if index in rates or not self._is_listed_unbanked(index):
                continue
            rate = float(rate)
            if not math.isfinite(rate) or rate <= 0.0:
                raise ValidationError(f"a recorded fixing rate must be positive and finite, got {rate}")
            rates[index] = rate
        if rates == self._recorded_rates:
            return self
        recorded = deepcopy(self)
        recorded._recorded_rates = rates
        return recorded

    def with_missing_fixings_filled(self, fx_env, *, warn: bool = True) -> "TargetRedemptionProduct":
        """This contract with every unbanked passed fixing's rate recorded at ``fx_env``'s QUOTED spot (Q3), on a
        COPY, so that bumped and shocked repricings bank it at the base spot. One ``MissingFixingWarning`` names the
        fixings filled (``warn=False`` skips it). Returns ``self`` when nothing is missing; never mutates."""
        filled = self._passed_fills(fx_env)
        if not filled:
            return self
        if warn:
            warn_filled_fixings(type(self).__name__, "a banked outcome", [label for _, label in filled])
        return self.with_fixings_recorded({index: fx_env.spot for index, _ in filled})

    def _is_listed_unbanked(self, index: int) -> bool:
        if self.is_numeric:
            return 0 <= index < len(self.fixing_times)
        return 0 <= index < len(self.fixing_dates) and not self.is_terminal and self._is_unbanked(index)

    def _has_unbanked_passed(self, fx_env) -> bool:
        if self.is_numeric:
            return bool(self.fixing_times) and is_passed_time(self.fixing_times[0])
        self._check_as_of(fx_env)
        if self.is_terminal:
            return False
        first = self._banked_count()  # fixings bank in schedule order: the first unbanked one decides
        return first < len(self.fixing_dates) and is_passed_time(
            signed_year_fraction(self.fixing_dates[first], fx_env))

    def _view(self, fx_env) -> ResolvedTargetRedemption:
        if self._recorded_rates and not self.is_terminal:  # every passed fixing is banked by now
            raise ValidationError(
                f"a fixing rate is recorded for future fixing(s) {sorted(self._recorded_rates)}: only a fixing at or "
                "before valuation has a rate")
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
        """Unbanked fixings (all ahead of valuation by time once ``resolve`` has banked the passed ones, OQ-2 b), their
        pay times and factors resolved against ``fx_env``; pending cash whose pay time is not strictly past (at exactly
        0.0 when due at the valuation instant, OQ-1 b, a weekend pay date under BUSINESS_DAYS included). Dead fixings
        of a redeemed contract are dropped."""
        st = self.state
        fix_t: List[float] = []
        pay_t: List[float] = []
        factors: List[float] = []
        if not self.is_terminal:
            for i in range(self._banked_count(), len(self.fixing_dates)):
                t = signed_year_fraction(self.fixing_dates[i], fx_env)
                fix_t.append(t)
                pay_t.append(signed_year_fraction(self.pay_dates[i], fx_env))
                if self._has_accrual:
                    factors.append(self._dated_factor(i, fx_env))
        pending: List[Tuple[float, float]] = []
        for p in st.pending:
            t = signed_year_fraction(p.pay_date, fx_env)
            if is_history(t):
                continue  # strictly past: paid, booked, not pending
            pending.append((snap_time(t), p.amount))
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
        """PV of the pending cash still due at or after valuation, cash due today at ``DF(0) = 1`` (no warning:
        callers have banked the passed fixings)."""
        return pending_present_value(self.resolve(fx_env, warn_missing_fixings=False).pending, fx_env)
