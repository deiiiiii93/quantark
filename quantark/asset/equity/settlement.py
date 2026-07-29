"""Shared equity cashflow settlement timing contract.

Products describe contractual payment terms.  Pricing engines consume the
normalized :class:`ResolvedPaymentTiming` returned here and must not duplicate
date derivation or payment discounting rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite
from typing import Optional, TYPE_CHECKING

from quantark.util.calendar import (
    BusinessDayConvention,
    Calendar,
    calculate_year_fraction,
)
from quantark.util.exceptions import ValidationError

if TYPE_CHECKING:
    from quantark.priceenv import PricingEnvironment


_TIME_TOLERANCE = 1.0e-10


class SettlementLagUnit(Enum):
    """Unit used to derive payment timing from determination timing."""

    BUSINESS_DAYS = "business_days"
    CALENDAR_DAYS = "calendar_days"
    YEAR_FRACTION = "year_fraction"


@dataclass(frozen=True)
class SettlementConvention:
    """Product-level rule for deriving a cashflow payment date or time."""

    lag: float = 0.0
    lag_unit: SettlementLagUnit = SettlementLagUnit.BUSINESS_DAYS
    business_day_convention: BusinessDayConvention = (
        BusinessDayConvention.FOLLOWING
    )
    calendar: Optional[Calendar] = None

    def __post_init__(self) -> None:
        try:
            lag = float(self.lag)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"settlement lag must be numeric, got {self.lag!r}"
            ) from exc

        if not isfinite(lag) or lag < 0.0:
            raise ValidationError(
                f"settlement lag must be finite and non-negative, got {self.lag!r}"
            )
        if not isinstance(self.lag_unit, SettlementLagUnit):
            raise ValidationError(
                f"invalid settlement lag unit: {self.lag_unit!r}"
            )
        if not isinstance(
            self.business_day_convention, BusinessDayConvention
        ):
            raise ValidationError(
                "invalid settlement business-day convention: "
                f"{self.business_day_convention!r}"
            )
        if self.calendar is not None and not isinstance(self.calendar, Calendar):
            raise ValidationError(
                f"settlement calendar must be Calendar, got {type(self.calendar).__name__}"
            )
        if (
            self.lag_unit
            in {
                SettlementLagUnit.BUSINESS_DAYS,
                SettlementLagUnit.CALENDAR_DAYS,
            }
            and not lag.is_integer()
        ):
            raise ValidationError(
                f"day-based settlement lag must be integral, got {self.lag!r}"
            )

        object.__setattr__(self, "lag", lag)


class CashflowKind(Enum):
    """Economic cashflow category used by settlement precedence rules."""

    TERMINAL = "terminal"
    EXERCISE = "exercise"
    HIT = "hit"
    OBSERVATION = "observation"
    COUPON = "coupon"
    REDEMPTION = "redemption"
    REBATE = "rebate"


@dataclass(frozen=True)
class SettlementRequest:
    """Raw determination and explicit payment timing for one cashflow."""

    kind: CashflowKind
    determination_date: Optional[datetime] = None
    determination_time: Optional[float] = None
    explicit_payment_date: Optional[datetime] = None
    explicit_payment_time: Optional[float] = None
    cashflow_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CashflowKind):
            raise ValidationError(f"invalid cashflow kind: {self.kind!r}")
        if self.cashflow_id is not None and not str(self.cashflow_id).strip():
            raise ValidationError("cashflow_id must be non-empty when provided")


@dataclass(frozen=True)
class ResolvedPaymentTiming:
    """Curve-ready determination and payment timing for one cashflow."""

    kind: CashflowKind
    determination_date: Optional[datetime]
    determination_time: float
    payment_date: Optional[datetime]
    payment_time: float
    determination_df: float
    payment_df: float
    delay_df: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CashflowKind):
            raise ValidationError(f"invalid resolved cashflow kind: {self.kind!r}")

        for name in ("determination_time", "payment_time"):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValidationError(f"{name} must be finite, got {value!r}")
            if value < -_TIME_TOLERANCE:
                raise ValidationError(
                    f"{name} must be non-negative for contingent valuation, got {value}"
                )

        if self.payment_time + _TIME_TOLERANCE < self.determination_time:
            raise ValidationError(
                "payment time cannot be before determination time"
            )

        for name in ("determination_df", "payment_df", "delay_df"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValidationError(
                    f"{name} must be finite and strictly positive, got {value!r}"
                )

        expected_delay = self.payment_df / self.determination_df
        if abs(self.delay_df - expected_delay) > _TIME_TOLERANCE:
            raise ValidationError(
                "delay_df is inconsistent with payment_df / determination_df"
            )


class SettlementResolver:
    """Resolve contractual settlement terms with one fail-closed precedence."""

    @classmethod
    def resolve_contingent(
        cls,
        product,
        request: SettlementRequest,
        pricing_env: "PricingEnvironment",
    ) -> ResolvedPaymentTiming:
        """Resolve timing for a live contingent cashflow.

        Determination must be at or after the pricing environment's valuation
        origin.  Pending realized cashflows use a separate resolver path added
        by the lifecycle integration.
        """

        if not isinstance(request, SettlementRequest):
            raise ValidationError(
                f"settlement request must be SettlementRequest, got {type(request).__name__}"
            )
        if pricing_env is None:
            raise ValidationError("pricing environment is required for settlement")

        context = cls._context(product, request)
        try:
            determination_date, determination_time = cls._resolve_pair(
                request.determination_date,
                request.determination_time,
                "determination",
                pricing_env,
            )
            if determination_time < -_TIME_TOLERANCE:
                raise ValidationError(
                    "determination time must be non-negative for contingent valuation"
                )
            determination_time = max(0.0, determination_time)

            payment_date, payment_time = cls._resolve_payment(
                product,
                request,
                determination_date,
                determination_time,
                pricing_env,
            )
            if payment_time + _TIME_TOLERANCE < determination_time:
                raise ValidationError(
                    "payment is before determination "
                    f"({payment_time:.12g} < {determination_time:.12g})"
                )

            determination_df = cls._checked_df(
                pricing_env, determination_time, "determination"
            )
            payment_df = cls._checked_df(
                pricing_env, payment_time, "payment"
            )

            return ResolvedPaymentTiming(
                kind=request.kind,
                determination_date=determination_date,
                determination_time=determination_time,
                payment_date=payment_date,
                payment_time=payment_time,
                determination_df=determination_df,
                payment_df=payment_df,
                delay_df=payment_df / determination_df,
            )
        except ValidationError as exc:
            if str(exc).startswith(context):
                raise
            raise ValidationError(f"{context}: {exc}") from exc

    @classmethod
    def _resolve_payment(
        cls,
        product,
        request: SettlementRequest,
        determination_date: Optional[datetime],
        determination_time: float,
        pricing_env: "PricingEnvironment",
    ) -> tuple[Optional[datetime], float]:
        if (
            request.explicit_payment_date is not None
            or request.explicit_payment_time is not None
        ):
            return cls._resolve_pair(
                request.explicit_payment_date,
                request.explicit_payment_time,
                "explicit payment",
                pricing_env,
            )

        terminal_settlement_date = getattr(product, "settlement_date", None)
        if (
            request.kind is CashflowKind.TERMINAL
            and terminal_settlement_date is not None
        ):
            return (
                terminal_settlement_date,
                cls._time_from_date(
                    terminal_settlement_date, pricing_env, "terminal settlement"
                ),
            )

        convention = getattr(product, "settlement_convention", None)
        if convention is None:
            return determination_date, determination_time
        if not isinstance(convention, SettlementConvention):
            raise ValidationError(
                "product settlement_convention must be SettlementConvention, "
                f"got {type(convention).__name__}"
            )

        if convention.lag == 0.0:
            return determination_date, determination_time

        if convention.lag_unit is SettlementLagUnit.YEAR_FRACTION:
            return None, determination_time + convention.lag

        if determination_date is None:
            raise ValidationError(
                f"{convention.lag_unit.name} settlement requires an "
                "authoritative determination date"
            )

        calendar = convention.calendar or getattr(pricing_env, "calendar", None)

        if convention.lag_unit is SettlementLagUnit.BUSINESS_DAYS:
            if calendar is None:
                raise ValidationError(
                    "business-day settlement requires a settlement or pricing calendar"
                )
            payment_date = calendar.add_business_days(
                determination_date, int(convention.lag)
            )
            payment_date = calendar.adjust_date(
                payment_date, convention.business_day_convention
            )
        elif convention.lag_unit is SettlementLagUnit.CALENDAR_DAYS:
            payment_date = determination_date + timedelta(
                days=int(convention.lag)
            )
            if (
                convention.business_day_convention
                is not BusinessDayConvention.UNADJUSTED
            ):
                if calendar is None:
                    raise ValidationError(
                        "calendar-day settlement adjustment requires a "
                        "settlement or pricing calendar"
                    )
                payment_date = calendar.adjust_date(
                    payment_date, convention.business_day_convention
                )
        else:
            raise ValidationError(
                f"unsupported settlement lag unit: {convention.lag_unit!r}"
            )

        return (
            payment_date,
            cls._time_from_date(payment_date, pricing_env, "derived payment"),
        )

    @classmethod
    def _resolve_pair(
        cls,
        date_value: Optional[datetime],
        time_value: Optional[float],
        label: str,
        pricing_env: "PricingEnvironment",
    ) -> tuple[Optional[datetime], float]:
        if date_value is None and time_value is None:
            raise ValidationError(
                f"{label} requires a date or numeric time representation"
            )

        time_from_date: Optional[float] = None
        if date_value is not None:
            time_from_date = cls._time_from_date(
                date_value, pricing_env, label
            )

        checked_time: Optional[float] = None
        if time_value is not None:
            try:
                checked_time = float(time_value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"{label} time must be numeric, got {time_value!r}"
                ) from exc
            if not isfinite(checked_time):
                raise ValidationError(
                    f"{label} time must be finite, got {time_value!r}"
                )

        if (
            time_from_date is not None
            and checked_time is not None
            and abs(time_from_date - checked_time) > _TIME_TOLERANCE
        ):
            raise ValidationError(
                f"{label} date and time are inconsistent: "
                f"{time_from_date:.12g} != {checked_time:.12g}"
            )

        resolved_time = (
            checked_time if checked_time is not None else time_from_date
        )
        assert resolved_time is not None
        return date_value, resolved_time

    @staticmethod
    def _time_from_date(
        date_value: datetime,
        pricing_env: "PricingEnvironment",
        label: str,
    ) -> float:
        if not isinstance(date_value, datetime):
            raise ValidationError(
                f"{label} date must be datetime, got {type(date_value).__name__}"
            )
        value = calculate_year_fraction(
            pricing_env.valuation_date,
            date_value,
            pricing_env.day_count_convention,
            pricing_env.bus_days_in_year,
            calendar=getattr(pricing_env, "calendar", None),
        )
        value = float(value)
        if not isfinite(value):
            raise ValidationError(
                f"{label} resolves to non-finite time {value!r}"
            )
        return value

    @staticmethod
    def _checked_df(
        pricing_env: "PricingEnvironment", time_value: float, label: str
    ) -> float:
        discount_factor = float(pricing_env.get_discount_factor(time_value))
        if not isfinite(discount_factor) or discount_factor <= 0.0:
            raise ValidationError(
                f"{label} discount factor must be finite and strictly positive, "
                f"got {discount_factor!r} at time {time_value:.12g}"
            )
        return discount_factor

    @staticmethod
    def _context(product, request: SettlementRequest) -> str:
        product_name = type(product).__name__
        cashflow_id = request.cashflow_id or "<unspecified>"
        convention = getattr(product, "settlement_convention", None)
        terminal_date = getattr(product, "settlement_date", None)
        return (
            f"{product_name} cashflow={request.kind.value} id={cashflow_id} "
            f"determination=(date={request.determination_date!r}, "
            f"time={request.determination_time!r}) "
            f"settlement=(explicit_date={request.explicit_payment_date!r}, "
            f"explicit_time={request.explicit_payment_time!r}, "
            f"terminal_date={terminal_date!r}, convention={convention!r})"
        )


__all__ = [
    "CashflowKind",
    "ResolvedPaymentTiming",
    "SettlementConvention",
    "SettlementLagUnit",
    "SettlementRequest",
    "SettlementResolver",
]
