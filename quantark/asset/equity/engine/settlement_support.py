"""Shared settlement timing operations and fail-closed engine guards."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from quantark.asset.equity.settlement import (
    CashflowKind,
    ResolvedPaymentTiming,
    SettlementRequest,
    SettlementResolver,
)
from quantark.execution.errors import CapabilityError
from quantark.util.exceptions import ValidationError

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
        raise CapabilityError(
            f"{type(engine).__name__} settlement support is "
            f"{declared.value}; request requires {required.value}"
        )
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
    "SettlementSupport",
    "apply_determination_to_payment",
    "pending_receivable_pv",
    "requested_settlement_support",
    "resolve_terminal_timing",
    "terminal_lifecycle_pv",
    "validate_settlement_capability",
]
