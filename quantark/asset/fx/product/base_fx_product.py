"""
Base class for FX derivative products.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from quantark.util.calendar import calculate_year_fraction
from quantark.util.exceptions import ValidationError
from .currency_pair import CurrencyPair

if TYPE_CHECKING:
    from quantark.priceenv import FxPricingEnvironment


class BaseFxProduct(ABC):
    """
    Abstract base class for all FX derivative products.

    FX products are pure instrument specifications on a currency pair: they
    carry no market data. Values are expressed in the domestic (quote)
    currency of the pair unless stated otherwise.

    Maturity may be given either as a year fraction (`maturity`) or as an
    `expiry_date` resolved against the pricing environment's valuation date
    and day count convention. An optional delivery lag (`delivery` years or
    `delivery_date`) defaults to the maturity when unset.

    Attributes:
        currency_pair: The currency pair of the product
        maturity: Time to expiry in years (mutually exclusive with expiry_date)
        expiry_date: Expiry date (mutually exclusive with maturity)
        delivery: Time to delivery/settlement in years (optional)
        delivery_date: Delivery/settlement date (optional)
    """

    def __init__(
        self,
        currency_pair: Optional[CurrencyPair] = None,
        maturity: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
        delivery: Optional[float] = None,
        delivery_date: Optional[datetime] = None,
    ):
        self.currency_pair = (
            currency_pair if currency_pair is not None else CurrencyPair()
        )
        self.maturity = maturity
        self.expiry_date = expiry_date
        self.delivery = delivery
        self.delivery_date = delivery_date

    @abstractmethod
    def get_payoff(self, spot: float) -> float:
        """
        Payoff of the product for a terminal spot rate, in domestic currency.

        Args:
            spot: FX rate at expiry (quote currency per base currency)

        Returns:
            Payoff value in domestic (quote) currency
        """

    @abstractmethod
    def validate(self) -> None:
        """
        Validate product parameters.

        Raises:
            ValidationError: If parameters are invalid
        """

    @property
    def is_linear(self) -> bool:
        """Return True for linear (delta-one) FX products."""
        return False

    def get_maturity(self, fx_env: Optional["FxPricingEnvironment"] = None) -> float:
        """
        Time to expiry in years from the valuation date.

        Uses the year-fraction `maturity` when set, otherwise resolves
        `expiry_date` against the pricing environment.

        Raises:
            ValidationError: If neither maturity nor expiry_date is usable
        """
        if self.maturity is not None:
            return self.maturity
        if self.expiry_date is None:
            raise ValidationError("Either maturity or expiry_date must be provided")
        return self._year_fraction_to(self.expiry_date, fx_env)

    def get_delivery(self, fx_env: Optional["FxPricingEnvironment"] = None) -> float:
        """
        Time to delivery/settlement in years from the valuation date.

        Defaults to the time to expiry when no delivery lag is specified.
        """
        if self.delivery is not None:
            return self.delivery
        if self.delivery_date is not None:
            return self._year_fraction_to(self.delivery_date, fx_env)
        return self.get_maturity(fx_env)

    def time_shift(self, time_bump: float) -> None:
        """
        Shift the product forward in time for theta bumping.

        Only supported for year-fraction maturities; date-based products are
        bumped by moving the environment valuation date instead.
        """
        if self.maturity is not None:
            self.maturity -= time_bump
        if self.delivery is not None:
            self.delivery -= time_bump

    def _year_fraction_to(
        self, target_date: datetime, fx_env: Optional["FxPricingEnvironment"]
    ) -> float:
        if fx_env is None:
            raise ValidationError(
                "FxPricingEnvironment required for date-based maturity calculation"
            )
        if fx_env.valuation_date > target_date:
            raise ValidationError(
                f"Valuation date ({fx_env.valuation_date}) must be on or before "
                f"target date ({target_date})"
            )
        if fx_env.valuation_date == target_date:
            return 0.0
        return calculate_year_fraction(
            fx_env.valuation_date,
            target_date,
            fx_env.day_count_convention,
            fx_env.bus_days_in_year,
            calendar=fx_env.calendar,
        )

    def _validate_maturity_inputs(self) -> None:
        """Shared maturity validation for concrete products."""
        if self.maturity is None and self.expiry_date is None:
            raise ValidationError("Either maturity or expiry_date must be provided")
        if self.maturity is not None and self.expiry_date is not None:
            raise ValidationError("Cannot provide both maturity and expiry_date")
        if self.maturity is not None and self.maturity <= 0:
            raise ValidationError(f"Maturity must be positive, got {self.maturity}")
        if self.delivery is not None and self.delivery_date is not None:
            raise ValidationError("Cannot provide both delivery and delivery_date")
        if (
            self.delivery is not None
            and self.maturity is not None
            and self.delivery < self.maturity
        ):
            raise ValidationError(
                f"Delivery ({self.delivery}) must be on or after expiry "
                f"({self.maturity})"
            )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.currency_pair})"
