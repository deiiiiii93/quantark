"""Shared settlement timing operations and fail-closed engine guards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence, TYPE_CHECKING

import numpy as np

from quantark.asset.equity.settlement import (
    CashflowKind,
    ResolvedPaymentTiming,
    SettlementLagUnit,
    SettlementRequest,
    SettlementResolver,
)
from quantark.execution.errors import CapabilityError
from quantark.util.calendar import calculate_year_fraction
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

from .capabilities import SettlementSupport

if TYPE_CHECKING:
    from quantark.asset.equity.lifecycle import EquityOptionLifecycleState
    from quantark.priceenv import PricingEnvironment


_SUPPORT_LEVEL = {
    SettlementSupport.NONE: 0,
    SettlementSupport.TERMINAL_ONLY: 1,
    SettlementSupport.EVENT_AND_TERMINAL: 2,
    SettlementSupport.AMERICAN_EXERCISE: 3,
}

_EVENT_PRODUCT_NAMES = {
    "AccumulatorOption",
    "BarrierOption",
    "DCNOption",
    "DoubleBarrierOption",
    "DoubleOneTouchOption",
    "DoubleSharkfinOption",
    "KnockOutResetSnowballOption",
    "OneTouchOption",
    "PhoenixOption",
    "RangeAccrualOption",
    "SingleSharkfinOption",
    "SnowballOption",
}


@dataclass(frozen=True)
class AmericanExerciseTimings:
    """Curve-ready payment timing carried by an American exercise grid.

    A date-based PDE may have diffusion nodes between contractual exercise
    dates. Those nodes remain in ``node_times`` but are marked ineligible and
    carry NaN payment data. Engines must apply the exercise obstacle only
    where ``eligible`` is true.
    """

    node_times: np.ndarray
    node_dates: tuple[Optional[datetime], ...]
    eligible: np.ndarray
    payment_dates: tuple[Optional[datetime], ...]
    payment_times: np.ndarray
    payment_dfs: np.ndarray
    delay_dfs: np.ndarray

    def __post_init__(self) -> None:
        node_times = np.array(self.node_times, dtype=float, copy=True)
        eligible = np.array(self.eligible, dtype=bool, copy=True)
        payment_times = np.array(self.payment_times, dtype=float, copy=True)
        payment_dfs = np.array(self.payment_dfs, dtype=float, copy=True)
        delay_dfs = np.array(self.delay_dfs, dtype=float, copy=True)
        node_dates = tuple(self.node_dates)
        payment_dates = tuple(self.payment_dates)
        size = node_times.size
        arrays = (
            eligible,
            payment_times,
            payment_dfs,
            delay_dfs,
        )
        if node_times.ndim != 1 or any(
            array.ndim != 1 or array.size != size for array in arrays
        ):
            raise ValidationError(
                "American exercise timing arrays must be one-dimensional "
                "and aligned"
            )
        if len(node_dates) != size or len(payment_dates) != size:
            raise ValidationError(
                "American exercise timing dates must align with node_times"
            )
        for array in (node_times, *arrays):
            array.setflags(write=False)
        object.__setattr__(self, "node_times", node_times)
        object.__setattr__(self, "eligible", eligible)
        object.__setattr__(self, "payment_times", payment_times)
        object.__setattr__(self, "payment_dfs", payment_dfs)
        object.__setattr__(self, "delay_dfs", delay_dfs)
        object.__setattr__(self, "node_dates", node_dates)
        object.__setattr__(self, "payment_dates", payment_dates)


def american_exercise_requires_dates(product) -> bool:
    """Whether the requested convention needs a real date at every exercise."""
    convention = getattr(product, "settlement_convention", None)
    if convention is None or float(convention.lag) == 0.0:
        return False
    return convention.lag_unit in {
        SettlementLagUnit.BUSINESS_DAYS,
        SettlementLagUnit.CALENDAR_DAYS,
    }


def build_american_exercise_date_grid(
    product,
    pricing_env: "PricingEnvironment",
) -> tuple[tuple[datetime, ...], np.ndarray]:
    """Enumerate authoritative calendar dates from valuation through expiry.

    Dates come only from the two contractual endpoints. No date is inferred
    from a numeric year fraction.
    """
    exercise_date = getattr(product, "exercise_date", None)
    if exercise_date is None:
        raise ValidationError(
            "day-based American settlement requires an authoritative "
            "exercise_date; a numeric American product may use only a "
            "YEAR_FRACTION settlement lag"
        )
    valuation_date = pricing_env.valuation_date
    if exercise_date < valuation_date:
        raise ValidationError(
            "American exercise_date cannot precede valuation_date"
        )
    if exercise_date == valuation_date:
        return (valuation_date,), np.array([0.0], dtype=float)

    dates = [valuation_date]
    current = valuation_date
    while current < exercise_date:
        current = min(current + timedelta(days=1), exercise_date)
        dates.append(current)

    resolved_dates: list[datetime] = []
    resolved_times: list[float] = []
    for date in dates:
        if date == valuation_date:
            time = 0.0
        else:
            time = float(
                calculate_year_fraction(
                    valuation_date,
                    date,
                    pricing_env.day_count_convention,
                    pricing_env.bus_days_in_year,
                    calendar=getattr(pricing_env, "calendar", None),
                )
            )
        if resolved_times and time <= resolved_times[-1]:
            # BUSINESS_DAYS time does not advance on weekends/holidays.
            # Such dates cannot be separate stochastic nodes on that clock.
            continue
        resolved_dates.append(date)
        resolved_times.append(time)

    if not resolved_dates or resolved_dates[-1] != exercise_date:
        raise ValidationError(
            "exercise_date does not form a distinct node under the pricing "
            "environment day-count convention"
        )

    maturity = float(product.get_maturity(pricing_env))
    if not is_close(resolved_times[-1], maturity):
        raise ValidationError(
            "American exercise date grid is inconsistent with product maturity"
        )
    resolved_times[-1] = maturity
    return tuple(resolved_dates), np.asarray(resolved_times, dtype=float)


def resolve_american_exercise_timings(
    product,
    pricing_env: "PricingEnvironment",
    node_times: Sequence[float],
    *,
    exercise_dates: Optional[Sequence[Optional[datetime]]] = None,
) -> AmericanExerciseTimings:
    """Resolve one delayed-exercise obstacle factor per eligible grid node.

    ``exercise_dates`` may contain ``None`` for diffusion-only PDE nodes.
    Date-based conventions reject a grid with no authoritative dates.
    """
    times = np.asarray(node_times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValidationError(
            "American exercise node_times must be a non-empty 1D array"
        )
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValidationError(
            "American exercise node_times must be finite and non-negative"
        )
    if np.any(np.diff(times) <= 0.0):
        raise ValidationError(
            "American exercise node_times must be strictly increasing"
        )

    if exercise_dates is None:
        dates: tuple[Optional[datetime], ...] = (None,) * times.size
    else:
        dates = tuple(exercise_dates)
        if len(dates) != times.size:
            raise ValidationError(
                "exercise_dates must align one-for-one with node_times"
            )
        if any(
            date is not None and not isinstance(date, datetime)
            for date in dates
        ):
            raise ValidationError(
                "American exercise node dates must be datetime or None"
            )

    requires_dates = american_exercise_requires_dates(product)
    if requires_dates and not any(date is not None for date in dates):
        raise ValidationError(
            "day-based American settlement requires an authoritative "
            "exercise_date and date-carrying exercise grid"
        )

    eligible = np.ones(times.size, dtype=bool)
    if requires_dates:
        eligible = np.asarray(
            [date is not None for date in dates],
            dtype=bool,
        )
        if not eligible[-1]:
            raise ValidationError(
                "date-based American settlement requires a terminal "
                "exercise-node date"
            )

    payment_dates: list[Optional[datetime]] = [None] * times.size
    payment_times = np.full(times.size, np.nan, dtype=float)
    payment_dfs = np.full(times.size, np.nan, dtype=float)
    delay_dfs = np.full(times.size, np.nan, dtype=float)

    for index, (node_time, node_date) in enumerate(zip(times, dates)):
        if not eligible[index]:
            continue
        kind = (
            CashflowKind.TERMINAL
            if index == times.size - 1
            else CashflowKind.EXERCISE
        )
        timing = SettlementResolver.resolve_contingent(
            product,
            SettlementRequest(
                kind=kind,
                determination_date=node_date,
                determination_time=(
                    None if node_date is not None else float(node_time)
                ),
                cashflow_id=f"american_exercise_{index}",
            ),
            pricing_env,
        )
        if timing.payment_time + 1.0e-10 < node_time:
            raise ValidationError(
                "American exercise payment cannot precede its numerical node"
            )
        node_df = float(pricing_env.get_discount_factor(float(node_time)))
        if not np.isfinite(node_df) or node_df <= 0.0:
            raise ValidationError(
                f"American exercise node DF must be positive at {node_time}"
            )
        payment_dates[index] = timing.payment_date
        payment_times[index] = timing.payment_time
        payment_dfs[index] = timing.payment_df
        delay_dfs[index] = timing.payment_df / node_df

    return AmericanExerciseTimings(
        node_times=times,
        node_dates=dates,
        eligible=eligible,
        payment_dates=tuple(payment_dates),
        payment_times=payment_times,
        payment_dfs=payment_dfs,
        delay_dfs=delay_dfs,
    )


def requested_settlement_support(
    product,
    lifecycle_state: Optional["EquityOptionLifecycleState"] = None,
) -> SettlementSupport:
    """Return the minimum support level required by one pricing request."""
    has_lifecycle = lifecycle_state is not None
    convention = getattr(product, "settlement_convention", None)
    convention_lag = 0.0
    if convention is not None:
        try:
            convention_lag = float(convention.lag)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValidationError(
                "product settlement_convention is malformed"
            ) from exc

    has_terminal_date = getattr(product, "settlement_date", None) is not None
    has_explicit_event_timing = _has_explicit_event_timing(product)
    requested = (
        has_lifecycle
        or convention_lag != 0.0
        or has_terminal_date
        or has_explicit_event_timing
    )
    if not requested:
        return SettlementSupport.NONE

    exercise_type = getattr(product, "exercise_type", None)
    if getattr(exercise_type, "name", None) == "AMERICAN":
        return SettlementSupport.AMERICAN_EXERCISE

    is_event_product = type(product).__name__ in _EVENT_PRODUCT_NAMES
    if (
        has_explicit_event_timing
        or (has_lifecycle and is_event_product)
        or (convention_lag != 0.0 and is_event_product)
    ):
        return SettlementSupport.EVENT_AND_TERMINAL
    return SettlementSupport.TERMINAL_ONLY


def validate_settlement_capability(
    engine,
    product,
    lifecycle_state: Optional["EquityOptionLifecycleState"] = None,
) -> SettlementSupport:
    """Reject unsupported timing before simulation or grid construction."""
    if lifecycle_state is not None and not getattr(
        engine, "supports_lifecycle_state", False
    ):
        raise CapabilityError(
            f"{type(engine).__name__} does not support lifecycle_state"
        )
    required = requested_settlement_support(product, lifecycle_state)
    declared = getattr(engine, "settlement_support", SettlementSupport.NONE)
    if not isinstance(declared, SettlementSupport):
        raise ValidationError(
            f"{type(engine).__name__}.settlement_support must be "
            f"SettlementSupport, got {declared!r}"
        )
    if _SUPPORT_LEVEL[declared] < _SUPPORT_LEVEL[required]:
        message = (
            f"{type(engine).__name__} settlement support is "
            f"{declared.value}; request requires {required.value}"
        )
        hint = getattr(engine, "settlement_capability_hint", None)
        if hint:
            message = f"{message}; {hint}"
        raise CapabilityError(message)
    return required


def resolve_terminal_timing(
    product,
    pricing_env: "PricingEnvironment",
) -> ResolvedPaymentTiming:
    """Resolve one terminal determination/payment pair."""
    exercise_date = getattr(product, "exercise_date", None)
    request = SettlementRequest(
        kind=CashflowKind.TERMINAL,
        determination_date=exercise_date,
        determination_time=(
            None if exercise_date is not None else product.get_maturity(pricing_env)
        ),
        cashflow_id="terminal",
    )
    return SettlementResolver.resolve_contingent(
        product,
        request,
        pricing_env,
    )


def apply_determination_to_payment(
    value_at_determination,
    timing: ResolvedPaymentTiming,
):
    """Move a value already discounted to determination on to payment."""
    if not isinstance(timing, ResolvedPaymentTiming):
        raise ValidationError(
            "timing must be ResolvedPaymentTiming, "
            f"got {type(timing).__name__}"
        )
    return value_at_determination * timing.delay_df


def pending_receivable_pv(
    lifecycle_state: Optional["EquityOptionLifecycleState"],
    pricing_env: "PricingEnvironment",
) -> float:
    """Return the PV of fixed, realized, not-yet-paid lifecycle cashflows."""
    if lifecycle_state is None:
        return 0.0
    ledger = getattr(lifecycle_state, "ledger", None)
    if ledger is None:
        raise ValidationError("lifecycle_state requires a cashflow ledger")
    valuation_point = getattr(lifecycle_state, "valuation_point", None)
    if valuation_point is None:
        if ledger.cashflows:
            raise ValidationError(
                "lifecycle_state with realized cashflows requires "
                "an explicit valuation_point"
            )
        return 0.0
    return ledger.pending_pv(
        valuation_point,
        pricing_env,
    )


def terminal_lifecycle_pv(
    lifecycle_state: Optional["EquityOptionLifecycleState"],
    pricing_env: "PricingEnvironment",
) -> Optional[float]:
    """Return fixed-cashflow PV when the contingent contract has terminated."""
    if lifecycle_state is None:
        return None

    terminal = (
        getattr(lifecycle_state, "alive", None) is False
        or bool(getattr(lifecycle_state, "matured", False))
        or bool(getattr(lifecycle_state, "expired", False))
        or bool(getattr(lifecycle_state, "knocked_out", False))
    )
    if not terminal:
        return None

    ledger = getattr(lifecycle_state, "ledger", None)
    if ledger is None or not ledger.cashflows:
        raise ValidationError(
            "terminal lifecycle_state requires an authoritative "
            "realized cashflow"
        )
    return pending_receivable_pv(lifecycle_state, pricing_env)


def _has_explicit_event_timing(product) -> bool:
    schedules = []
    direct = getattr(product, "observation_schedule", None)
    if direct is not None:
        schedules.append(direct)

    barrier_config = getattr(product, "barrier_config", None)
    if barrier_config is not None:
        for name in ("ko_observation_schedule", "ki_observation_schedule"):
            schedule = getattr(barrier_config, name, None)
            if schedule is not None:
                schedules.append(schedule)

    post_barrier_config = getattr(product, "post_barrier_config", None)
    if post_barrier_config is not None:
        schedule = getattr(
            post_barrier_config, "ko_observation_schedule", None
        )
        if schedule is not None:
            schedules.append(schedule)

    for schedule in schedules:
        for record in getattr(schedule, "records", ()):
            if (
                getattr(record, "settlement_date", None) is not None
                or getattr(record, "settlement_time", None) is not None
            ):
                return True
    return False


__all__ = [
    "AmericanExerciseTimings",
    "SettlementSupport",
    "american_exercise_requires_dates",
    "apply_determination_to_payment",
    "build_american_exercise_date_grid",
    "pending_receivable_pv",
    "requested_settlement_support",
    "resolve_american_exercise_timings",
    "resolve_terminal_timing",
    "terminal_lifecycle_pv",
    "validate_settlement_capability",
]
