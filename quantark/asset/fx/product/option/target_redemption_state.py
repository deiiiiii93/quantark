"""Realized state of an FX target-redemption contract (TARN / TARF).

Spec: docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md, section 3. ``accumulated`` is the undiscounted
amount banked towards the target (TARN coupons, TARF favourable amounts, including those fixed but not yet paid).
``pending`` is determined cash whose pay time is after valuation. The state is immutable: ageing replaces it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Final, Optional, Sequence, Tuple

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero

#: Relative target-hit tolerance, moved from the two MC engines' ``_HIT_RTOL`` with its value unchanged (spec section
#: 3: ``Tolerance.RELATIVE`` would move redemption decisions and inception prices).
TARGET_HIT_RTOL: Final[float] = 1e-12


def target_hit_tol(target: float) -> float:
    """Target-relative hit tolerance; 0 for an unbounded target (the engines' rule, unchanged)."""
    return 0.0 if math.isinf(target) else abs(target) * TARGET_HIT_RTOL


def is_target_hit(accumulated: float, target: float) -> bool:
    """True once ``accumulated`` reaches ``target`` within :func:`target_hit_tol`."""
    return accumulated >= target - target_hit_tol(target)


@dataclass(frozen=True)
class PendingPayment:
    """Determined, unpaid domestic cash: ``pay_time`` (numeric form, years from valuation) or ``pay_date`` (dated)."""

    amount: float
    pay_time: Optional[float] = None
    pay_date: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.amount):
            raise ValidationError(f"pending amount must be finite, got {self.amount}")
        if (self.pay_time is None) == (self.pay_date is None):
            raise ValidationError(
                "a pending payment takes exactly one of pay_time (numeric form) or pay_date (dated form)"
            )
        if self.pay_time is not None:
            t = self.pay_time
            if not math.isfinite(t) or t <= 0.0 or is_zero(t):
                raise ValidationError(
                    f"a numeric pending pay_time must be after valuation (> 0), got {t}: cash paid at or before "
                    "valuation is booked, not pending"
                )


@dataclass(frozen=True)
class TargetRedemptionState:
    """What a target-redemption contract has banked (``accumulated``) and still has to pay (``pending``).

    ``as_of`` is used by the dated form only: every fixing dated on or before it is banked.
    """

    accumulated: float
    pending: Tuple[PendingPayment, ...] = ()
    as_of: Optional[datetime] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pending", tuple(self.pending))
        acc = self.accumulated
        if not math.isfinite(acc) or acc < 0.0:
            raise ValidationError(f"state.accumulated must be finite and >= 0, got {acc}")
        if any(not isinstance(p, PendingPayment) for p in self.pending):
            raise ValidationError("state.pending holds PendingPayment entries")
        numeric = [p.pay_time is not None for p in self.pending]
        if any(numeric) and not all(numeric):
            raise ValidationError("state.pending mixes pay_time and pay_date entries: one form per contract")
        keys = [p.pay_time if p.pay_time is not None else p.pay_date for p in self.pending]
        if keys != sorted(keys):
            raise ValidationError("state.pending must be sorted by ascending pay time / date")
        if self.pending and not any(numeric):
            if self.as_of is None:
                raise ValidationError(
                    "dated pending cash with as_of=None is refused: cash implies a banked fixing"
                )
            late = [p.pay_date for p in self.pending if p.pay_date <= self.as_of]
            if late:
                raise ValidationError(f"a dated pending pay_date must be after as_of {self.as_of}, got {late[0]}")


class TerminalBookingWarning(UserWarning):
    """A contract was BOOKED terminal (redeemed, or every fixing banked): it is valued at the PV of its pending cash,
    0.0 when none (spec section 5.3, user decision Q6). Raised once, by the constructor."""


class TargetRedemptionEventType(Enum):
    FIXING = "fixing"  # a fixing was banked
    PAYMENT = "payment"  # determined cash reached its pay time / date


@dataclass(frozen=True)
class TargetRedemptionEvent:
    """One lifecycle event, appended to a product's ``lifecycle_events`` by ``time_shift`` / ``record_passed_fixings``.

    ``amount``: FIXING - the cash it determined (coupon / net cashflow, principal excluded); PAYMENT - the cash paid.
    ``time``: FIXING - the fixing's signed time at banking (<= 0); PAYMENT - the pay time (<= 0).
    ``redeemed``: FIXING - this fixing ended the contract (target hit, or the last fixing).
    ``label``: W5 ``fixing_label`` text; an immediate PAYMENT carries its FIXING's label (informational).
    """

    event_type: TargetRedemptionEventType
    amount: float
    fixing: Optional[float] = None
    time: float = 0.0
    redeemed: bool = False
    label: str = ""


@dataclass(frozen=True)
class ResolvedTargetRedemption:
    """The one engine input (spec section 5.1): unbanked fixings still ahead (all > 0), their factors (TARN), the
    banked amount, pending ``(pay_time, amount)`` after valuation, and the derived flags."""

    fixing_times: Tuple[float, ...]
    pay_times: Tuple[float, ...]
    accrual_factors: Optional[Tuple[float, ...]]
    accumulated: float
    pending: Tuple[Tuple[float, float], ...]
    redeemed: bool
    terminal: bool


def pending_present_value(pending: Sequence[Tuple[float, float]], fx_env) -> float:
    """``sum(amount * DF_dom(pay_time))`` over ``(pay_time, amount)`` pairs, in order."""
    pv = 0.0
    for pay_time, amount in pending:
        pv += amount * fx_env.get_domestic_df(pay_time)
    return pv
