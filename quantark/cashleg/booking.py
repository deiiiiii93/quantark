"""Multi-day cash-leg booking (spec 2026-10-23 portfolio-trade-value D11–D14).

A position's cash legs carry year fractions measured from the run's start date. Each run day ``d`` the book
rolls the ORIGINAL legs by ``τ_d = whole days / 365`` from that anchor (the frame lifecycle trackers, the float
roller and the ledger already use; no accumulated drift) and pays, once and at nominal, every flow the shared
landing predicate (:func:`quantark.cashleg.base.roll_payment_time`) says is no longer ahead.

While the position lives only flows that need no termination outcome pay: a ``DeterministicLeg`` amount and an
``AccrualLeg`` period paid on its own schedule (the note being alive, every past survival reference survived).
On termination every unpaid flow becomes a fixed receivable by an explicit rule per leg (:class:`RealizedOutcome`),
revalued each day until it pays. The rules are not ``leg.value`` on a degenerate distribution: survival there is
1 at and before the valuation instant and interpolated between observations, while a realized path is a step at
the knock-out observation. They agree with ``leg.value`` on a complete probability-1 distribution whose survival
references sit on the observation grid.

Cases with no specified realized rule raise ``CapabilityError`` rather than guess (spec C6): a ``FixedPayoffLeg``
on knock-in, a ``NOTIONAL_MINUS_PAYOFF`` or ``COUPON``-basis ``AutocallableCashLeg``, an ``AT_KO`` accrual.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date as _date
from datetime import datetime
from enum import Enum
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from quantark.cashleg.accrual_leg import AccrualLeg, KOBehavior, PaymentConvention, SurvivalBasis
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, PvFormula
from quantark.cashleg.base import CashLeg, roll_payment_time
from quantark.cashleg.deterministic_leg import DeterministicLeg
from quantark.cashleg.event_distribution import EventType
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.cashleg.leg_schedule import LegSchedule
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import almost_equal, is_zero

DAYS_PER_YEAR = 365.0

#: (leg_id, flow index) — one flow of one leg
FlowKey = Tuple[str, int]
#: (key, signed nominal amount, payment time in the run frame)
Flow = Tuple[FlowKey, float, float]


def _day(value) -> _date:
    return value.date() if isinstance(value, datetime) or hasattr(value, "date") else value


def run_time(anchor, date) -> float:
    """Run-frame time of ``date``: whole days from ``anchor`` over 365 (time of day ignored)."""
    return (_day(date) - _day(anchor)).days / DAYS_PER_YEAR


class OutcomeKind(Enum):
    """How the position ended."""

    KO = "ko"
    MATURITY = "maturity"


@dataclass(frozen=True)
class RealizedOutcome:
    """The realized termination of a position, times in the run frame.

    ``event_time`` is the deciding observation (KO) and ``settlement_time`` its payment; ``maturity_settlement_time``
    is the contractual maturity settlement (where ``AT_MATURITY`` accruals and maturity-triggered payoffs pay).
    """

    kind: OutcomeKind
    event_time: float
    settlement_time: float
    maturity_settlement_time: float
    knocked_in: bool = False

    @classmethod
    def ko(cls, event_time: float, settlement_time: float, *, maturity_settlement_time: float) -> "RealizedOutcome":
        """Knocked out at the observation ``event_time``, its redemption paid at ``settlement_time``."""
        return cls(OutcomeKind.KO, float(event_time), float(settlement_time), float(maturity_settlement_time))

    @classmethod
    def maturity(cls, knocked_in: bool, settlement_time: float) -> "RealizedOutcome":
        """Reached maturity (knocked in or not), settled at ``settlement_time``."""
        t = float(settlement_time)
        return cls(OutcomeKind.MATURITY, t, t, t, bool(knocked_in))


@dataclass(frozen=True)
class LegBookDay:
    """One run day: leg cash paid that day, the receivables' PV, and the live (rolled) legs."""

    paid_cash: float
    receivable_pv: float
    live_legs: Tuple[CashLeg, ...]


_SUPPORTED = (DeterministicLeg, AccrualLeg, FixedPayoffLeg, AutocallableCashLeg)


def _refusal(leg: CashLeg) -> Optional[str]:
    if type(leg) not in _SUPPORTED:
        return f"{type(leg).__name__} has no multi-day booking rule"
    if isinstance(leg, FixedPayoffLeg) and leg.trigger is PaymentTrigger.AT_KI:
        return "a FixedPayoffLeg on knock-in has no realized payment rule"
    if isinstance(leg, AutocallableCashLeg):
        if leg.pv_formula is not PvFormula.NORMAL:
            return "a NOTIONAL_MINUS_PAYOFF AutocallableCashLeg has no realized settlement rule for its notional"
        if leg.accrual_basis is not AccrualBasis.KO_MATURITY:
            return "a COUPON-basis AutocallableCashLeg reads realized coupon events the book does not track"
    if isinstance(leg, AccrualLeg) and leg.payment_convention is PaymentConvention.AT_KO:
        return "an AT_KO AccrualLeg has no specified realized payment time"
    if (isinstance(leg, AccrualLeg) and leg.payment_convention is PaymentConvention.AT_PERIOD_END
            and leg.ko_behavior is KOBehavior.TRUNCATE_AT_KO):
        s = leg.schedule
        refs = s.period_starts if leg.survival_basis is SurvivalBasis.ENTER_PERIOD else s.period_ends
        early = [i for i, (t, r) in enumerate(zip(s.payment_times, refs))
                 if float(t) < float(r) and not is_zero(float(t) - float(r))]
        if early:
            return (f"periods {early} pay before their survival reference; a later knock-out could not "
                    "reverse cash already paid")
    return None


def _pays_only_at_termination(leg: CashLeg) -> bool:
    if isinstance(leg, AccrualLeg):
        return leg.payment_convention is PaymentConvention.AT_MATURITY
    return isinstance(leg, (FixedPayoffLeg, AutocallableCashLeg))


def _snap_leg(leg: CashLeg, snap: Callable[[float], float]) -> CashLeg:
    """The leg with its own times snapped (the tracker's rule), amounts unchanged."""
    if isinstance(leg, DeterministicLeg):
        return replace(leg, payment_time=snap(leg.payment_time))
    if isinstance(leg, AccrualLeg):
        s = leg.schedule
        fractions = None if s.has_period_dates() else tuple(float(f) for f in leg.accrual_fractions())
        schedule = LegSchedule(
            period_starts=np.array([snap(float(t)) for t in s.period_starts]),
            period_ends=np.array([snap(float(t)) for t in s.period_ends]),
            payment_times=np.array([snap(float(t)) for t in s.payment_times]),
            period_start_dates=s.period_start_dates,
            period_end_dates=s.period_end_dates,
            payment_dates=s.payment_dates,
        )
        return replace(leg, schedule=schedule, fixed_fractions=fractions)
    if isinstance(leg, AutocallableCashLeg):
        return replace(
            leg,
            observation_schedule=tuple(snap(float(t)) for t in leg.observation_schedule),
            settlement_schedule=tuple(snap(float(t)) for t in leg.settlement_schedule),
            terminal_settlement_time=snap(float(leg.terminal_settlement_time)),
        )
    return leg


def _accrual_amounts(leg: AccrualLeg, unit_notional: float) -> np.ndarray:
    return leg.sign() * float(leg.rate) * float(leg.base.resolve(unit_notional)) * leg.accrual_fractions()


def _live_flows(leg: CashLeg, unit_notional: float) -> List[Tuple[int, float, float]]:
    """Flows payable while the position lives: ``(index, amount, payment time)``."""
    if isinstance(leg, DeterministicLeg):
        return [(0, leg.sign() * float(leg.amount), float(leg.payment_time))]
    if isinstance(leg, AccrualLeg) and leg.payment_convention is PaymentConvention.AT_PERIOD_END:
        amounts = _accrual_amounts(leg, unit_notional)
        return [(i, float(a), float(t)) for i, (a, t) in enumerate(zip(amounts, leg.schedule.payment_times))]
    return []


def _termination_flows(leg: CashLeg, outcome: RealizedOutcome, unit_notional: float) -> List[Tuple[int, float, float]]:
    """Every flow of ``leg`` under ``outcome``: ``(index, amount, payment time)``; flows that do not pay omitted."""
    ko = outcome.kind is OutcomeKind.KO
    if isinstance(leg, DeterministicLeg):
        return _live_flows(leg, unit_notional)
    if isinstance(leg, AccrualLeg):
        s = leg.schedule
        refs = s.period_starts if leg.survival_basis is SurvivalBasis.ENTER_PERIOD else s.period_ends
        truncate = ko and leg.ko_behavior is KOBehavior.TRUNCATE_AT_KO
        flows = []
        for i, amount in enumerate(_accrual_amounts(leg, unit_notional)):
            # the engine prices survival after the observation: a reference ON the KO date did not survive
            if truncate and not (float(refs[i]) < outcome.event_time and not is_zero(float(refs[i]) - outcome.event_time)):
                continue
            pay = (outcome.maturity_settlement_time if leg.payment_convention is PaymentConvention.AT_MATURITY
                   else float(s.payment_times[i]))
            flows.append((i, float(amount), pay))
        return flows
    if isinstance(leg, FixedPayoffLeg):
        offset = float(leg.payment_offset_days) / DAYS_PER_YEAR
        amount = leg.sign() * float(leg.amount)
        if leg.trigger is PaymentTrigger.AT_KO:
            return [(0, amount, outcome.settlement_time + offset)] if ko else []
        if ko:
            return []
        wanted = {
            PaymentTrigger.AT_MATURITY_NO_KO: not outcome.knocked_in,
            PaymentTrigger.AT_MATURITY_WITH_KI: outcome.knocked_in,
            PaymentTrigger.AT_MATURITY_ANY: True,
        }[leg.trigger]
        return [(0, amount, outcome.maturity_settlement_time + offset)] if wanted else []
    if isinstance(leg, AutocallableCashLeg):
        scale = leg.sign() * float(leg.notional) * float(leg.rate)
        if ko:
            obs = np.asarray(leg.observation_schedule, dtype=float)
            if obs.size == 0:
                return []
            k = int(np.argmin(np.abs(obs - outcome.event_time)))
            if abs(float(obs[k]) - outcome.event_time) > 0.5 / DAYS_PER_YEAR:
                from quantark.execution.errors import CapabilityError

                raise CapabilityError(
                    f"AutocallableCashLeg {leg.leg_id}: the knock-out at t={outcome.event_time:.6f} is not one of "
                    "its observations; align the leg to the note's schedule"
                )
            return [(k, scale * float(leg.accrual_factors[k]), float(leg.settlement_schedule[k]))]
        bucket = EventType.MATURITY_WITH_KI if outcome.knocked_in else EventType.MATURITY_NO_KO
        if bucket not in leg.terminal_events:
            return []
        return [(len(leg.observation_schedule), scale * float(leg.terminal_accrual_factor),
                 float(leg.terminal_settlement_time))]
    raise ValidationError(f"no termination rule for {type(leg).__name__}")


class LegCashBook:
    """One position's cash legs through a multi-day run (spec 2026-10-23 D12).

    Args:
        legs: The position's cash legs, times in years from ``anchor_date``.
        unit_notional: The position's per-unit notional that fraction-based amounts resolve against.
        anchor_date: The legs' time origin: the contract's own clock origin (its ``initial_date`` when set, the
            lifecycle tracker's origin), else the run's start date (user ruling 2026-10-23).
        snap: The lifecycle tracker's time snap for a tracked position (D14), applied to every leg time.
        terminable: False for a position no lifecycle tracker ends; legs that pay only at termination are
            then refused.
        run_start: The run's start date when it is later than ``anchor_date`` (a trade replayed mid-life): flows
            paid strictly before it are history — neither booked nor valued; the run sees every later flow's
            time to payment from the run start.

    Raises:
        CapabilityError: A leg with no specified realized rule, or a termination-paying leg on a
            non-terminable position.
    """

    def __init__(
        self,
        legs: Sequence[CashLeg],
        unit_notional: float,
        anchor_date,
        *,
        snap: Optional[Callable[[float], float]] = None,
        terminable: bool = True,
        run_start=None,
    ):
        from quantark.execution.errors import CapabilityError

        for leg in legs:
            reason = _refusal(leg)
            if reason is None and not terminable and _pays_only_at_termination(leg):
                reason = (f"{type(leg).__name__} pays only when the note ends, and no lifecycle tracker ends "
                          "this position")
            if reason is not None:
                raise CapabilityError(f"multi-day cash-leg booking: {reason} (leg {leg.leg_id})")
        ids = [leg.leg_id for leg in legs]
        if len(set(ids)) != len(ids):
            raise ValidationError(f"cash-leg book: legs must have distinct leg_id values, got {ids}")
        self.legs: Tuple[CashLeg, ...] = tuple(_snap_leg(leg, snap) if snap else leg for leg in legs)
        self.unit_notional = float(unit_notional)
        self.anchor_date = anchor_date
        self.terminated = False
        #: the flows paid by the last ``roll`` / ``terminate`` call
        self.paid_today: List[Tuple[FlowKey, float]] = []
        self._paid: set = set()
        self._receivables: List[Flow] = []
        self._tau: Optional[float] = None
        #: flows paid before the run started (history: never booked)
        self.history: List[Tuple[FlowKey, float]] = []
        self.run_start_time = 0.0 if run_start is None else max(0.0, run_time(anchor_date, run_start))
        if self.run_start_time > 0.0:
            for leg in self.legs:
                for i, amount, t in _live_flows(leg, self.unit_notional):
                    if roll_payment_time(t, self.run_start_time, keep_landing=True) is None:   # strictly before
                        self._paid.add((leg.leg_id, i))
                        self.history.append(((leg.leg_id, i), amount))

    @property
    def unpaid_receivables(self) -> float:
        """Nominal of the termination receivables not paid yet (0 before termination)."""
        return float(sum(a for key, a, _t in self._receivables if key not in self._paid))

    @property
    def settled(self) -> bool:
        """True once the position has ended and every termination receivable has paid."""
        return self.terminated and all(key in self._paid for key, _a, _t in self._receivables)

    def legs_at_run_start(self) -> Tuple[CashLeg, ...]:
        """The legs as the run's first (pre-close) valuation sees them: rolled to the run start, a flow due that
        day kept at 0 (it pays on the run's first close)."""
        rolled = (leg.time_shift(self.run_start_time, keep_landing=True) for leg in self.legs)
        return tuple(leg for leg in rolled if leg is not None)

    def _advance(self, date) -> float:
        tau = run_time(self.anchor_date, date)
        if self._tau is not None and tau < self._tau:
            raise ValidationError(f"cash-leg book cannot go back in time (τ {tau} after {self._tau})")
        self._tau = tau
        self.paid_today = []
        return tau

    def _pay(self, flows: Iterable[Flow], tau: float) -> float:
        paid = 0.0
        for key, amount, t in flows:
            if key not in self._paid and roll_payment_time(t, tau) is None:
                self._paid.add(key)
                self.paid_today.append((key, amount))
                paid += amount
        return paid

    def _receivable_pv(self, env, tau: float) -> float:
        return float(sum(a * env.get_discount_factor(t - tau)
                         for key, a, t in self._receivables if key not in self._paid))

    def roll(self, date, env) -> LegBookDay:
        """Advance to ``date``: pay the flows now due; the live legs rolled to it (none once terminated)."""
        tau = self._advance(date)
        if self.terminated:
            paid = self._pay(self._receivables, tau)
            return LegBookDay(paid, self._receivable_pv(env, tau), ())
        live = [((leg.leg_id, i), a, t) for leg in self.legs for i, a, t in _live_flows(leg, self.unit_notional)]
        paid = self._pay(live, tau)
        rolled = (leg.time_shift(tau) for leg in self.legs)
        return LegBookDay(paid, 0.0, tuple(leg for leg in rolled if leg is not None))

    def termination_flows(self, outcome: RealizedOutcome) -> List[Flow]:
        """The unpaid flows ``outcome`` makes payable, in the run frame (zero amounts omitted)."""
        flows = []
        for leg in self.legs:
            for i, amount, t in _termination_flows(leg, outcome, self.unit_notional):
                key = (leg.leg_id, i)
                if key not in self._paid and not almost_equal(amount, 0.0, tol=0.0):
                    flows.append((key, amount, t))
        return flows

    def terminate(self, outcome: RealizedOutcome, date, env) -> LegBookDay:
        """The position ended on ``date``: every unpaid flow becomes a receivable under ``outcome``; those already
        due pay today. Called instead of ``roll`` on that day, so a flow due today is decided once, by the outcome."""
        if self.terminated:
            raise ValidationError("cash-leg book already terminated")
        tau = self._advance(date)
        self._receivables = self.termination_flows(outcome)
        self.terminated = True
        paid = self._pay(self._receivables, tau)
        return LegBookDay(paid, self._receivable_pv(env, tau), ())
