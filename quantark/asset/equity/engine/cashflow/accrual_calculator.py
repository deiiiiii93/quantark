"""Interest-accrual strategies for the Total Return Swap fixed leg.

Each strategy computes simple interest over a calendar-day count: ``notional *
rate * days / basis``. The basis (``act/365`` or ``act/360``) divides the
calendar-day count; the day count itself comes from the calendar's
``get_num_of_calendar_days`` (head/tail controlled by ``side``).
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from quantark.util.exceptions import ValidationError
from quantark.asset.equity.product.swap.trs_params import AccrualType, AccrualSide

_INTEREST_BASIS_DAYS = {"act/365": 365, "act/360": 360}


def _resolve_basis_days(day_count_basis: str) -> int:
    """Return the denominator for ``day_count_basis`` or raise on an unknown value."""
    key = day_count_basis.lower() if isinstance(day_count_basis, str) else day_count_basis
    if key not in _INTEREST_BASIS_DAYS:
        raise ValidationError(
            f"Unsupported day_count_basis: {day_count_basis!r}; "
            f"expected one of {sorted(_INTEREST_BASIS_DAYS)}"
        )
    return _INTEREST_BASIS_DAYS[key]


def _normalize_side(side: Any) -> str:
    """Accept an ``AccrualSide`` enum or a string and return the calendar side string."""
    return side.value if isinstance(side, AccrualSide) else side


class AccrualCalculator(ABC):
    """Abstract base for interest-accrual strategies."""

    @abstractmethod
    def calculate_accrual(
        self,
        notional: float,
        start_date: str,
        end_date: str,
        interest_rate: float,
        calendar: Any,
        side: str = "left",
        day_count_basis: str = "act/365",
        force_no_zero: bool = False,
        **kwargs: Any,
    ) -> float:
        """Return the interest accrued over ``[start_date, end_date]``."""
        raise NotImplementedError


class StandardAccrualCalculator(AccrualCalculator):
    """Simple-interest accrual on an explicit notional."""

    def calculate_accrual(
        self,
        notional: float,
        start_date: str,
        end_date: str,
        interest_rate: float,
        calendar: Any,
        side: str = "left",
        day_count_basis: str = "act/365",
        force_no_zero: bool = False,
        **kwargs: Any,
    ) -> float:
        accrual_days = calendar.get_num_of_calendar_days(
            start_date, end_date, side=_normalize_side(side)
        )
        if force_no_zero:
            accrual_days = max(accrual_days, 1)
        basis = _resolve_basis_days(day_count_basis)
        return notional * interest_rate * accrual_days / basis


class NotionalAccrualCalculator(StandardAccrualCalculator):
    """Accrual on a fixed notional (identical to the standard strategy)."""


class MarketValueAccrualCalculator(AccrualCalculator):
    """Accrual on the current market value (``asset_quantity * asset_price``)."""

    def calculate_accrual(
        self,
        notional: float,
        start_date: str,
        end_date: str,
        interest_rate: float,
        calendar: Any,
        side: str = "left",
        day_count_basis: str = "act/365",
        force_no_zero: bool = False,
        asset_quantity: Optional[float] = None,
        asset_price: Optional[float] = None,
        **kwargs: Any,
    ) -> float:
        if asset_quantity is None or asset_price is None:
            raise ValidationError(
                "asset_quantity and asset_price must be provided for "
                "MarketValueAccrualCalculator"
            )
        market_value = asset_quantity * asset_price
        return StandardAccrualCalculator().calculate_accrual(
            market_value, start_date, end_date, interest_rate, calendar,
            side, day_count_basis, force_no_zero,
        )


class LastMarketValueAccrualCalculator(AccrualCalculator):
    """Accrual on the previous period's market value."""

    def calculate_accrual(
        self,
        notional: float,
        start_date: str,
        end_date: str,
        interest_rate: float,
        calendar: Any,
        side: str = "left",
        day_count_basis: str = "act/365",
        force_no_zero: bool = False,
        last_asset_quantity: Optional[float] = None,
        last_asset_price: Optional[float] = None,
        **kwargs: Any,
    ) -> float:
        if last_asset_quantity is None or last_asset_price is None:
            raise ValidationError(
                "last_asset_quantity and last_asset_price must be provided for "
                "LastMarketValueAccrualCalculator"
            )
        last_market_value = last_asset_quantity * last_asset_price
        return StandardAccrualCalculator().calculate_accrual(
            last_market_value, start_date, end_date, interest_rate, calendar,
            side, day_count_basis, force_no_zero,
        )


class AccrualCalculatorFactory:
    """Factory mapping :class:`AccrualType` to a concrete strategy."""

    @staticmethod
    def create_calculator(accrual_type: AccrualType) -> AccrualCalculator:
        if accrual_type == AccrualType.NOTIONAL:
            return NotionalAccrualCalculator()
        if accrual_type == AccrualType.MARKET_VALUE:
            return MarketValueAccrualCalculator()
        if accrual_type == AccrualType.LAST_MARKET_VALUE:
            return LastMarketValueAccrualCalculator()
        raise ValidationError(f"Unknown accrual type: {accrual_type!r}")
