"""
Pricing environment that bundles all market data.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from param import SpotQuote, VolatilitySurface, RateCurve, DividendYield
from util.exceptions import MarketDataError
from util.calendar import DayCountConvention


@dataclass
class PricingEnvironment:
    """
    Pricing environment containing all market data required for pricing.
    
    This class bundles together all market parameters needed for derivative pricing:
    spot price, volatility surface, risk-free rate curve, and dividend yield.
    
    Attributes:
        spot_quote: Current spot price of the underlying
        vol_surface: Volatility surface
        rate_curve: Risk-free rate curve
        div_yield: Dividend yield (optional, defaults to zero)
        valuation_date: Date of valuation (required)
        day_count_convention: Convention for calculating year fractions (default: CALENDAR_DAYS)
        bus_days_in_year: Number of business days per year for business day convention (default: 252)
    """
    spot_quote: SpotQuote
    vol_surface: VolatilitySurface
    rate_curve: RateCurve
    div_yield: Optional[DividendYield] = None
    valuation_date: datetime = None
    day_count_convention: DayCountConvention = DayCountConvention.CALENDAR_DAYS
    bus_days_in_year: int = 252
    
    def __post_init__(self):
        """Validate pricing environment."""
        if self.spot_quote is None:
            raise MarketDataError("Spot quote is required")
        if self.vol_surface is None:
            raise MarketDataError("Volatility surface is required")
        if self.rate_curve is None:
            raise MarketDataError("Rate curve is required")
        if self.valuation_date is None:
            raise MarketDataError("Valuation date is required")
        if self.bus_days_in_year <= 0:
            raise MarketDataError(f"Business days per year must be positive, got {self.bus_days_in_year}")
    
    @property
    def spot(self) -> float:
        """Get current spot price."""
        return self.spot_quote.spot
    
    def get_vol(self, strike: float, time_to_maturity: float) -> float:
        """
        Get volatility for given strike and maturity.
        
        Args:
            strike: Strike price
            time_to_maturity: Time to maturity in years
            
        Returns:
            Implied volatility
        """
        return self.vol_surface.get_vol(strike, time_to_maturity, self.spot)
    
    def get_rate(self, time_to_maturity: float) -> float:
        """
        Get risk-free rate for given maturity.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Risk-free rate
        """
        return self.rate_curve.get_rate(time_to_maturity)
    
    def get_discount_factor(self, time_to_maturity: float) -> float:
        """
        Get discount factor for given maturity.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Discount factor
        """
        return self.rate_curve.get_discount_factor(time_to_maturity)
    
    def get_div_yield(self, time_to_maturity: float) -> float:
        """
        Get dividend yield for given maturity.
        
        Args:
            time_to_maturity: Time to maturity in years
            
        Returns:
            Dividend yield (0 if not specified)
        """
        if self.div_yield is None:
            return 0.0
        return self.div_yield.get_yield(time_to_maturity)
    
    def __repr__(self):
        return (f"PricingEnvironment(spot={self.spot:.2f}, "
                f"rate={self.rate_curve.get_rate(1.0):.2%}, "
                f"vol={self.vol_surface.get_vol(self.spot, 1.0, self.spot):.2%})")

