"""Shared machinery of the FX target-redemption products (TARN and TARF).

Spec: docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md, sections 3-6. A contract carries its realized
state (``state``: the amount banked towards the target and the determined cash still to pay). In the NUMERIC form
``fixing_times`` / ``pay_times`` are years from valuation and list ONLY the fixings not yet banked; a time at or
before valuation is an unbanked past fixing. Only the two concrete products are instantiated.
"""
from __future__ import annotations

import math
import warnings
from datetime import datetime
from types import SimpleNamespace
from typing import List, Optional, Tuple

from quantark.util.exceptions import ValidationError
from quantark.util.fixings import fixing_label
from quantark.util.numerical import is_zero

from ..base_fx_product import BaseFxProduct
from .target_redemption_state import (
    PendingPayment,
    TargetRedemptionEvent,
    TargetRedemptionEventType,
    TargetRedemptionState,
    TerminalBookingWarning,
    is_target_hit,
    target_hit_tol,
)


def is_passed_time(t: float) -> bool:
    """A numeric fixing or pay time at or before valuation; ``is_zero`` absorbs the ULPs daily ageing leaves."""
    return t <= 0.0 or is_zero(t)


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
        return not self.fixing_times or self.is_redeemed

    # -- validation ---------------------------------------------------------------------------------------------
    def validate(self) -> None:
        self._validate_terms()
        self._validate_numeric_schedule()
        self._validate_state()

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
        if st.as_of is not None:
            raise ValidationError("state.as_of belongs to the dated form: a numeric state lists only unbanked fixings")
        if any(p.pay_date is not None for p in st.pending):
            raise ValidationError("a numeric contract's pending payments take pay_time, not pay_date")
        if self.is_redeemed and self.fixing_times:
            raise ValidationError("a redeemed state cannot list fixings: every fixing after the target hit is dead")

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
        stopping once the contract is terminal; append the events; re-validate."""
        self._record_numeric(fx_env)
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
