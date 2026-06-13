"""
FX pricing environment bundling market data for a single currency pair.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from quantark.param import RateCurve, SpotQuote, VolatilitySurface
from quantark.util.calendar import Calendar, DayCountConvention
from quantark.util.exceptions import MarketDataError
from quantark.util.numerical import safe_exp


@dataclass
class FxQuantoMarketData:
    """
    Market data required for quanto FX option pricing.

    A quanto FX option's payoff is determined by one currency pair but settled
    in a third currency at a contractually fixed conversion rate. Pricing
    requires the settlement currency's discount curve, the volatility of the
    conversion pair, and the correlation between the underlying pair and the
    conversion pair.

    Note: the fixed contractual conversion rate (quanto_fx_rate) is a product
    attribute, not market data — it lives on the quanto product.

    Attributes:
        settlement_curve: Discount curve of the settlement currency
        quanto_vol: Volatility of the conversion (quanto) currency pair
        correlation: Correlation between underlying pair and conversion pair
    """

    settlement_curve: RateCurve
    quanto_vol: float
    correlation: float

    def __post_init__(self):
        if self.settlement_curve is None:
            raise MarketDataError("Quanto settlement curve is required")
        if self.quanto_vol < 0:
            raise MarketDataError(
                f"Quanto volatility must be non-negative, got {self.quanto_vol}"
            )
        if not -1.0 <= self.correlation <= 1.0:
            raise MarketDataError(
                f"Quanto correlation must be in [-1, 1], got {self.correlation}"
            )


@dataclass
class FxPricingEnvironment:
    """
    Pricing environment for FX derivatives on a single currency pair.

    The spot quote follows the quote-per-base convention: for EUR/USD a spot
    of 1.20 means 1 EUR = 1.20 USD. In Garman-Kohlhagen terms the quote
    currency is the *domestic* (pricing) currency and the base currency is
    the *foreign* (asset) currency; option values are expressed in the
    domestic currency.

    Attributes:
        valuation_date: Date of valuation (required)
        spot_quote: Current FX spot rate, quote currency per base currency (required)
        domestic_curve: Quote-currency (domestic) risk-free curve (required)
        foreign_curve: Base-currency (foreign) risk-free curve (required)
        vol_surface: FX volatility surface (optional; required by option engines)
        market_forward: Market-quoted outright forward overriding interest-rate
            parity when set (optional)
        spot_days: Spot settlement lag in days. When positive, effective_spot()
            grows the quoted (overnight) rate to the spot date using the two
            curves; default 0 means the quote is already the spot-date rate.
        quanto: Optional quanto market data for quanto products
        day_count_convention: Convention for converting dates to year fractions
        bus_days_in_year: Business days per year for business-day convention
        calendar: Optional business day calendar
    """

    valuation_date: datetime
    spot_quote: SpotQuote
    domestic_curve: RateCurve
    foreign_curve: RateCurve
    vol_surface: Optional[VolatilitySurface] = None
    market_forward: Optional[float] = None
    spot_days: int = 0
    quanto: Optional[FxQuantoMarketData] = None
    day_count_convention: DayCountConvention = DayCountConvention.CALENDAR_DAYS
    bus_days_in_year: int = 252
    calendar: Optional[Calendar] = None

    def __post_init__(self):
        """Validate the pricing environment."""
        if self.valuation_date is None:
            raise MarketDataError("Valuation date is required")
        if self.spot_quote is None:
            raise MarketDataError("FX spot quote is required")
        if self.domestic_curve is None:
            raise MarketDataError("Domestic (quote currency) rate curve is required")
        if self.foreign_curve is None:
            raise MarketDataError("Foreign (base currency) rate curve is required")
        if self.spot_days < 0:
            raise MarketDataError(
                f"Spot settlement lag must be non-negative, got {self.spot_days}"
            )
        if self.bus_days_in_year <= 0:
            raise MarketDataError(
                f"Business days per year must be positive, got {self.bus_days_in_year}"
            )

    @property
    def spot(self) -> float:
        """Current FX spot rate (quote currency per base currency)."""
        return self.spot_quote.spot

    def get_domestic_rate(self, time_to_maturity: float) -> float:
        """Zero rate of the domestic (quote) currency for the given maturity."""
        return self.domestic_curve.get_rate(time_to_maturity)

    def get_foreign_rate(self, time_to_maturity: float) -> float:
        """Zero rate of the foreign (base) currency for the given maturity."""
        return self.foreign_curve.get_rate(time_to_maturity)

    def get_domestic_df(self, time_to_maturity: float) -> float:
        """Discount factor of the domestic (quote) currency."""
        return self.domestic_curve.get_discount_factor(time_to_maturity)

    def get_foreign_df(self, time_to_maturity: float) -> float:
        """Discount factor of the foreign (base) currency."""
        return self.foreign_curve.get_discount_factor(time_to_maturity)

    def get_vol(self, strike: float, time_to_maturity: float) -> float:
        """
        FX volatility for the given strike and maturity.

        Raises:
            MarketDataError: If no volatility surface is available
        """
        if self.vol_surface is None:
            raise MarketDataError(
                "Volatility surface not available in this FX pricing environment"
            )
        return self.vol_surface.get_vol(strike, time_to_maturity, self.spot)

    def effective_spot(self) -> float:
        """
        Spot rate adjusted from the quoted (overnight) rate to the spot date.

        When spot_days is zero the quote is used as-is. Otherwise the quoted
        rate is grown over the settlement lag using interest rate parity:
            S_eff = S * exp((r_d - r_f) * tau_spot)
        """
        if self.spot_days == 0:
            return self.spot
        tau_spot = self._spot_lag_year_fraction()
        r_d = self.get_domestic_rate(tau_spot)
        r_f = self.get_foreign_rate(tau_spot)
        return self.spot * float(safe_exp((r_d - r_f) * tau_spot))

    def get_forward(self, time_to_maturity: float) -> float:
        """
        Outright forward rate for the given maturity.

        Returns the market-quoted forward when one is set; otherwise computes
        the interest-rate-parity forward from the effective spot:
            F = S_eff * df_foreign(T) / df_domestic(T)

        Raises:
            MarketDataError: If time to maturity is negative
        """
        if time_to_maturity < 0:
            raise MarketDataError(
                f"Time to maturity must be non-negative, got {time_to_maturity}"
            )
        if self.market_forward is not None:
            return self.market_forward
        return (
            self.effective_spot()
            * self.get_foreign_df(time_to_maturity)
            / self.get_domestic_df(time_to_maturity)
        )

    def _spot_lag_year_fraction(self) -> float:
        """Year fraction of the spot settlement lag under the env's convention."""
        if self.day_count_convention == DayCountConvention.BUSINESS_DAYS:
            return self.spot_days / self.bus_days_in_year
        return self.spot_days / 365.0

    def __repr__(self):
        return (
            f"FxPricingEnvironment(valuation_date={self.valuation_date.date()}, "
            f"spot={self.spot:.6f}, "
            f"r_dom={self.get_domestic_rate(1.0):.2%}, "
            f"r_for={self.get_foreign_rate(1.0):.2%})"
        )
