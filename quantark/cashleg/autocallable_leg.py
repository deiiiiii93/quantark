"""Autocallable-driven cash legs (margin/backend/rebate/minimum-return).

These legs are contingent on the same KO (or Phoenix coupon) observation schedule
as the parent autocallable. They consume the parent's ``EventDistribution`` and
apply a per-observation accrual schedule with explicit, workbook-sourced
settlement times. Greeks flow through the existing position bump loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from quantark.cashleg.base import CashLeg
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import (
    Tolerance,
    almost_equal,
    is_valid_number,
    validate_positive,
)


class AutocallableLegType(Enum):
    """Label for an autocallable adjustment cash leg (no hidden behaviour)."""

    MARGIN = "margin"
    BACKEND_PREMIUM = "backend_premium"
    BACKEND_INTEREST = "backend_interest"
    REBATE = "rebate"
    MINIMUM_RETURN = "minimum_return"


class PvFormula(Enum):
    """How the discounted contingent return R maps to leg PV."""

    NORMAL = "normal"                                # PV = sign * R
    NOTIONAL_MINUS_PAYOFF = "notional_minus_payoff"  # PV = sign * (notional - R)


class AccrualBasis(Enum):
    """Which probability stream drives the contingent accruals."""

    KO_MATURITY = "ko_maturity"   # KO observation dates + terminal branch
    COUPON = "coupon"             # Phoenix coupon observations + terminal branch


_TERMINAL_BUCKETS = frozenset(
    {EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI}
)
_BASIS_EVENT = {
    AccrualBasis.KO_MATURITY: EventType.KO,
    AccrualBasis.COUPON: EventType.COUPON,
}


@dataclass(frozen=True)
class AutocallableCashLeg(CashLeg):
    """KO/coupon-contingent return leg priced off a parent EventDistribution."""

    leg_type: Optional[AutocallableLegType] = None
    notional: float = 0.0
    rate: float = 0.0
    observation_schedule: Sequence[float] = ()
    accrual_factors: Sequence[float] = ()
    settlement_schedule: Sequence[float] = ()
    terminal_accrual_factor: float = 0.0
    terminal_settlement_time: float = 0.0
    pv_formula: PvFormula = PvFormula.NORMAL
    accrual_basis: AccrualBasis = AccrualBasis.KO_MATURITY
    terminal_events: frozenset = field(default_factory=lambda: _TERMINAL_BUCKETS)
    notional_settlement_time: Optional[float] = None

    def __post_init__(self) -> None:
        if self.leg_type is None or not isinstance(self.leg_type, AutocallableLegType):
            raise ValidationError("AutocallableCashLeg requires a valid leg_type")
        if not isinstance(self.pv_formula, PvFormula):
            raise ValidationError(f"Invalid PvFormula: {self.pv_formula}")
        if not isinstance(self.accrual_basis, AccrualBasis):
            raise ValidationError(f"Invalid AccrualBasis: {self.accrual_basis}")

        # notional is absolute per-unit => finite and non-negative.
        validate_positive(self.notional, "notional", allow_zero=True)
        if not is_valid_number(self.rate):
            raise ValidationError(f"rate must be finite, got {self.rate}")
        if not is_valid_number(self.terminal_accrual_factor):
            raise ValidationError("terminal_accrual_factor must be finite")
        if not is_valid_number(self.terminal_settlement_time) or self.terminal_settlement_time < 0.0:
            raise ValidationError("terminal_settlement_time must be finite and >= 0")

        n = len(self.observation_schedule)
        if not (len(self.accrual_factors) == n and len(self.settlement_schedule) == n):
            raise ValidationError(
                "observation_schedule, accrual_factors, settlement_schedule "
                f"must share one length; got {n}, {len(self.accrual_factors)}, "
                f"{len(self.settlement_schedule)}"
            )

        for label, seq in (
            ("observation_schedule", self.observation_schedule),
            ("accrual_factors", self.accrual_factors),
            ("settlement_schedule", self.settlement_schedule),
        ):
            arr = np.asarray(seq, dtype=float)
            if arr.size and not np.all(np.isfinite(arr)):
                raise ValidationError(f"{label} must be all-finite, got {seq!r}")
        obs_arr = np.asarray(self.observation_schedule, dtype=float)
        ss_arr = np.asarray(self.settlement_schedule, dtype=float)
        if np.any(obs_arr < 0.0) or np.any(ss_arr < 0.0):
            raise ValidationError(
                "observation_schedule and settlement_schedule times must be >= 0"
            )

        if not self.terminal_events or not self.terminal_events.issubset(_TERMINAL_BUCKETS):
            raise ValidationError(
                "terminal_events must be a non-empty subset of "
                "{MATURITY_NO_KO, MATURITY_WITH_KI}"
            )

        if self.pv_formula is PvFormula.NOTIONAL_MINUS_PAYOFF:
            if self.notional_settlement_time is not None:
                t0 = float(self.notional_settlement_time)
                if not is_valid_number(t0) or t0 > Tolerance.PROBABILITY:
                    raise ValidationError(
                        "NOTIONAL_MINUS_PAYOFF requires the notional to be an "
                        "outstanding claim as of valuation (notional_settlement_time "
                        f"<= 0); got {self.notional_settlement_time}. Use a "
                        "DeterministicLeg for a future-dated notional exchange."
                    )

    def requires_event_distribution(self) -> bool:
        return True

    def value(
        self, event_dist: EventDistribution, env, position_notional: float
    ) -> float:
        """Return signed PV from the buyer's perspective (uses self.notional)."""

        af = np.asarray(self.accrual_factors, dtype=float)
        contingent = 0.0
        if af.size:
            event_type = _BASIS_EVENT[self.accrual_basis]
            prob = event_dist.probabilities.get(event_type)
            if not isinstance(prob, np.ndarray):
                raise ValidationError(
                    f"AutocallableCashLeg ({self.accrual_basis.value}) requires an "
                    f"array {event_type.value} stream in the parent EventDistribution; "
                    "the parent engine emitted none (trivial/unsupported distribution)."
                )
            obs = np.asarray(self.observation_schedule, dtype=float)
            ss = np.asarray(self.settlement_schedule, dtype=float)
            ev_times = np.asarray(event_dist.event_times, dtype=float)
            if prob.shape != obs.shape:
                raise ValidationError(
                    f"{event_type.value} length {prob.shape} != observation_schedule "
                    f"length {obs.shape}; align the leg to the parent's future grid."
                )
            if obs.shape != ev_times.shape or not all(
                almost_equal(float(a), float(b), tol=Tolerance.PROBABILITY)
                for a, b in zip(obs, ev_times)
            ):
                raise ValidationError(
                    "observation_schedule does not match the parent's filtered "
                    "event_times (shifted/dropped observation); refusing to realign."
                )
            df_obs = np.array([env.get_discount_factor(float(t)) for t in ss])
            contingent = float(np.sum(af * prob * df_obs))

        for e in self.terminal_events:
            prob_e = event_dist.probabilities.get(e)
            if prob_e is None or isinstance(prob_e, np.ndarray) or not is_valid_number(float(prob_e)):
                raise ValidationError(
                    f"terminal event {e.value} missing or non-scalar in the parent "
                    "EventDistribution; cannot value the terminal branch."
                )
        p_term = sum(float(event_dist.probabilities[e]) for e in self.terminal_events)
        df_term = env.get_discount_factor(float(self.terminal_settlement_time))
        contingent += float(self.terminal_accrual_factor) * p_term * float(df_term)

        R = float(self.notional) * float(self.rate) * contingent
        if self.pv_formula is PvFormula.NORMAL:
            return self.sign() * R
        return self.sign() * (float(self.notional) - R)

    def value_standalone(self, parent_product, engine, env) -> float:
        """Price this leg directly against a parent product + engine."""

        result = engine.price_with_events(parent_product, env, emit_distribution=True)
        return self.value(result.event_distribution, env, float(self.notional))
