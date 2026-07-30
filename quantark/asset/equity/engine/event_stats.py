"""
Engine-level event stats and cashflow decomposition types.

These types define an optional API that engines MAY implement to provide
per-observation event probabilities and expected discounted cashflows for
autocallable products (Snowball-first).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

from quantark.util.exceptions import ValidationError


_PAYMENT_AWARE_ARRAY_FIELDS = (
    "determination_times",
    "payment_times",
    "expected_undiscounted_cashflows",
    "expected_discounted_cashflows",
)


def _normalize_payment_aware_fields(instance) -> None:
    """Copy, validate, and freeze one additive expected-cashflow ledger."""
    arrays = {
        name: np.array(getattr(instance, name), dtype=float, copy=True)
        for name in _PAYMENT_AWARE_ARRAY_FIELDS
    }
    sizes = {array.size for array in arrays.values()}
    if any(array.ndim != 1 for array in arrays.values()) or len(sizes) != 1:
        raise ValidationError(
            "payment-aware event-stat arrays must be one-dimensional and aligned"
        )
    size = arrays["determination_times"].size
    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValidationError(
            "payment-aware event-stat arrays must contain only finite values"
        )
    if size and np.any(
        arrays["payment_times"] + 1.0e-10
        < arrays["determination_times"]
    ):
        raise ValidationError(
            "event-stat payment times cannot precede determination times"
        )

    for name in ("determination_dates", "payment_dates"):
        dates = getattr(instance, name)
        if dates is None:
            continue
        dates = tuple(dates)
        if len(dates) != size or any(
            not isinstance(value, datetime) for value in dates
        ):
            raise ValidationError(
                f"{name} must be an aligned tuple of datetime values"
            )
        object.__setattr__(instance, name, dates)

    for name, array in arrays.items():
        array.setflags(write=False)
        object.__setattr__(instance, name, array)


def payment_aware_cashflow_fields(
    pricing_env,
    *,
    determination_times: Sequence[float],
    payment_times: Sequence[float],
    expected_discounted_cashflows: Sequence[float],
    determination_dates: Optional[Sequence[datetime]] = None,
    payment_dates: Optional[Sequence[datetime]] = None,
) -> dict:
    """Build curve-exact aligned expected-cashflow fields.

    Expected undiscounted cashflows are recovered from deterministic payment
    discount factors. Engines retain their probability grids separately;
    duplicate determination times are valid when different cashflow legs
    (for example Phoenix KO and coupon) share one observation.
    """
    determination = np.asarray(determination_times, dtype=float)
    payment = np.asarray(payment_times, dtype=float)
    discounted = np.asarray(expected_discounted_cashflows, dtype=float)
    if (
        determination.ndim != 1
        or payment.ndim != 1
        or discounted.ndim != 1
        or determination.size != payment.size
        or payment.size != discounted.size
    ):
        raise ValidationError(
            "expected cashflow determination, payment, and value arrays "
            "must be one-dimensional and aligned"
        )
    if np.any(~np.isfinite(determination)) or np.any(~np.isfinite(payment)):
        raise ValidationError(
            "expected cashflow determination and payment times must be finite"
        )
    if np.any(~np.isfinite(discounted)):
        raise ValidationError(
            "expected discounted cashflows must be finite"
        )
    if np.any(payment + 1.0e-10 < determination):
        raise ValidationError(
            "expected cashflow payment cannot precede determination"
        )

    payment_dfs = np.asarray(
        [
            pricing_env.get_discount_factor(float(time))
            for time in payment
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(payment_dfs)) or np.any(payment_dfs <= 0.0):
        raise ValidationError(
            "expected cashflow payment discount factors must be positive"
        )

    def _complete_dates(values):
        if values is None:
            return None
        resolved = tuple(values)
        if len(resolved) != determination.size:
            raise ValidationError(
                "expected cashflow dates must align with cashflow arrays"
            )
        if any(value is None for value in resolved):
            return None
        if any(not isinstance(value, datetime) for value in resolved):
            raise ValidationError(
                "expected cashflow dates must contain datetime values"
            )
        return resolved

    return {
        "determination_times": determination,
        "payment_times": payment,
        "expected_undiscounted_cashflows": discounted / payment_dfs,
        "expected_discounted_cashflows": discounted,
        "determination_dates": _complete_dates(determination_dates),
        "payment_dates": _complete_dates(payment_dates),
    }


@dataclass(frozen=True)
class AutocallableEventStats:
    """
    Event probabilities and expected discounted cashflows.

    Attributes:
        pv: Present value produced by the engine for the given product/env.
        ko_times: KO observation times (year fractions from valuation date).
        ko_probability: Probability of KO occurring at each observation time.
        survival_probability: Probability of surviving (not KO'd) up to each observation.
        expected_discounted_ko_cashflow: Expected discounted KO redemption cashflow at each observation.
        ki_probability: LEGACY KI probability whose *definition differs by engine* — kept
            for backward compatibility. QUAD/PDE report P(KI ever AND never KO)
            (the KI indicator is absorbed to 0 on any KO), while MC reports
            P(KI ever). Prefer the two unambiguous, cross-engine-consistent fields
            below; ``ki_probability`` is retained only so existing callers do not
            break.
        expected_discounted_maturity_cashflow: Expected discounted maturity cashflow (conditional on no KO).
        reconciliation_error: pv minus sum(expected discounted cashflows) if computed, else 0.0.
        ki_times: KI observation/monitoring times where event probabilities are available.
        ki_event_probability: Probability of first KI occurring at each KI time.
        ki_survival_probability: Probability of surviving without KI up to each KI time.
        ki_ever_probability: P(the KI barrier is breached at any point in the
            path's life, regardless of any subsequent KO/autocall). A *monitoring*
            statistic. ``None`` if the engine does not compute it.
        ki_survive_knocked_in_probability: P(KI breached AND the path reaches
            maturity without ever knocking out) = P(the note settles in the
            knocked-in state). This is the *economically relevant* quantity for
            downside exposure / loss distribution (a path that knocks in and then
            recovers to autocall redeems at par, so it carries no KI loss). Equals
            ``ki_ever_probability`` minus P(KI AND KO). ``None`` if not computed.
    """

    pv: float
    ko_times: np.ndarray
    ko_probability: np.ndarray
    survival_probability: np.ndarray
    expected_discounted_ko_cashflow: np.ndarray
    ki_probability: float
    expected_discounted_maturity_cashflow: float
    reconciliation_error: float = 0.0
    ki_times: np.ndarray = field(default_factory=lambda: np.array([]))
    ki_event_probability: np.ndarray = field(default_factory=lambda: np.array([]))
    ki_survival_probability: np.ndarray = field(default_factory=lambda: np.array([]))
    ki_ever_probability: Optional[float] = None
    ki_survive_knocked_in_probability: Optional[float] = None
    determination_times: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    payment_times: np.ndarray = field(default_factory=lambda: np.array([]))
    expected_undiscounted_cashflows: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    expected_discounted_cashflows: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    determination_dates: Optional[tuple[datetime, ...]] = None
    payment_dates: Optional[tuple[datetime, ...]] = None

    def __post_init__(self) -> None:
        _normalize_payment_aware_fields(self)


@dataclass(frozen=True)
class PhoenixEventStats(AutocallableEventStats):
    """
    Event stats for Phoenix options including coupon diagnostics.

    Attributes:
        coupon_probability: Coupon trigger probability at each observation time.
        expected_discounted_coupon_cashflow: Expected discounted coupon cashflow per observation.
    """

    coupon_probability: np.ndarray = field(default_factory=lambda: np.array([]))
    expected_discounted_coupon_cashflow: np.ndarray = field(
        default_factory=lambda: np.array([])
    )


@dataclass(frozen=True)
class KOResetEventStats(AutocallableEventStats):
    """
    Event stats for KO-reset snowball options.

    Attributes:
        pre_ko_times: Pre-KI KO observation times (absolute).
        pre_ko_probability: KO probability per pre-KI observation.
        post_ko_times: Post-KI KO observation times (absolute or offsets for REBASED).
        post_ko_probability: KO probability per post-KI observation (or offset).
        pre_ko_probability_total: Total probability of pre-KI KO.
        post_ko_probability_total: Total probability of post-KI KO.
        expected_discounted_post_ko_cashflow: Total expected discounted KO cashflow after KI.
    """

    pre_ko_times: np.ndarray = field(default_factory=lambda: np.array([]))
    pre_ko_probability: np.ndarray = field(default_factory=lambda: np.array([]))
    post_ko_times: np.ndarray = field(default_factory=lambda: np.array([]))
    post_ko_probability: np.ndarray = field(default_factory=lambda: np.array([]))
    pre_ko_probability_total: float = 0.0
    post_ko_probability_total: float = 0.0
    expected_discounted_post_ko_cashflow: float = 0.0


@dataclass(frozen=True)
class DCNEventStats:
    """Unconditional Q-measure event statistics for a DCN (spec WP1.3).

    Attributes:
        ki_probability: P(KI at any monitored date after valuation, or seeded).
        ko_probability: P(KO at any KO observation).
        ko_timing_distribution: Unconditional P(KO at obs j); sums to
            ko_probability.
        coupon_probability: P(fixed coupon paid at period j); KO coupon
            excluded (visible via ko_timing_distribution).
        expected_life_years: E[min(tau_KO_obs, T_last_obs)] in ACT/365F years
            measured at observation (not payment) dates.
        prob_survive_no_ki: P(alive at maturity and never knocked in).
        prob_survive_ki: P(alive at maturity and knocked in).
        expected_discounted_loss_leg: Signed expected discounted loss-leg PV.
    """

    ki_probability: float
    ko_probability: float
    ko_timing_distribution: tuple
    coupon_probability: tuple
    expected_life_years: float
    prob_survive_no_ki: float
    prob_survive_ki: float
    expected_discounted_loss_leg: float
    pv: float = 0.0
    determination_times: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    payment_times: np.ndarray = field(default_factory=lambda: np.array([]))
    expected_undiscounted_cashflows: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    expected_discounted_cashflows: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    determination_dates: Optional[tuple[datetime, ...]] = None
    payment_dates: Optional[tuple[datetime, ...]] = None

    def __post_init__(self) -> None:
        _normalize_payment_aware_fields(self)

    def to_dict(self) -> dict:
        def _json_value(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, tuple):
                return [_json_value(item) for item in value]
            if isinstance(value, datetime):
                return value.isoformat()
            return value

        return {k: _json_value(v) for k, v in self.__dict__.items()}


__all__ = [
    "AutocallableEventStats",
    "DCNEventStats",
    "KOResetEventStats",
    "PhoenixEventStats",
    "payment_aware_cashflow_fields",
]
