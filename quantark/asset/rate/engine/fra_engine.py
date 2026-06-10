"""
Analytical pricing engine for Forward Rate Agreements (FRA).

Pricing formula:
    FRA NPV = N * dcf * (L - K) / (1 + L * dcf) * df(T_settle)

where:
    N     = notional
    dcf   = day count fraction for the accrual period
    L     = forward rate for the accrual period
    K     = agreed FRA rate (fixed rate)
    df(T) = discount factor to settlement date

The settlement amount is discounted to the settlement date (accrual start),
then further discounted to the valuation date.

For a buyer (receiver of floating):
    - Positive NPV when forward rate > fixed rate
    - Negative NPV when forward rate < fixed rate
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from asset.rate.product.fra import ForwardRateAgreement
from priceenv import PricingEnvironment
from param.rrf import RateCurve, FlatRateCurve
from util.exceptions import ValidationError, MarketDataError
from util.numerical import safe_divide


@dataclass
class FRAPricingResults:
    """
    Results container for FRA pricing.

    Attributes:
        npv: Net present value of the FRA (from buyer's perspective)
        forward_rate: Implied forward rate for the accrual period
        settlement_amount: Undiscounted settlement amount at settlement date
        settlement_pv: Settlement amount discounted to valuation date
        day_count_fraction: Day count fraction for the accrual period
        dv01: Dollar value of 1 basis point parallel shift
        par_rate: Forward rate (fair FRA rate that makes NPV = 0)
    """

    npv: float
    forward_rate: float
    settlement_amount: float
    settlement_pv: float
    day_count_fraction: float
    dv01: Optional[float] = None
    par_rate: Optional[float] = None


class FRAEngine:
    """
    Analytical pricing engine for Forward Rate Agreements.

    Uses the standard FRA settlement formula with discounted cashflows.
    The engine supports single-curve pricing (same curve for discounting
    and forward rate projection) or dual-curve pricing with a separate
    projection curve.
    """

    def __init__(
        self,
        pricing_env: PricingEnvironment,
        projection_curve: Optional[RateCurve] = None,
    ):
        """
        Initialize the FRA engine.

        Args:
            pricing_env: Pricing environment with discount curve
            projection_curve: Optional separate curve for forward rate projection.
                             If None, uses the discount curve from pricing_env.
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")

        if pricing_env.rate_curve is None:
            raise MarketDataError("Rate curve is required for FRA pricing")

        self.pricing_env = pricing_env
        self.projection_curve = projection_curve or pricing_env.rate_curve

    def price(
        self,
        fra: ForwardRateAgreement,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate the NPV of a FRA (from buyer's perspective).

        The buyer pays fixed and receives floating. Positive NPV means
        the forward rate exceeds the fixed rate.

        Args:
            fra: Forward Rate Agreement to price
            valuation_date: Valuation date (default: pricing env date)

        Returns:
            Net present value
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if fra.is_expired(valuation_date):
            return 0.0

        # Calculate forward rate for the accrual period
        forward_rate = self._get_forward_rate(fra, valuation_date)

        # Day count fraction for the accrual period
        dcf = fra.day_count_fraction()

        # Settlement amount (at settlement date, discounted within period)
        # FRA settlement = N * dcf * (L - K) / (1 + L * dcf)
        rate_diff = forward_rate - fra.fixed_rate
        denominator = 1.0 + forward_rate * dcf
        settlement_amount = safe_divide(
            fra.notional * dcf * rate_diff, denominator, fallback=0.0
        )

        # Discount from settlement date to valuation date
        t_settle = fra.time_to_settlement(valuation_date)
        df = self.pricing_env.get_discount_factor(t_settle)

        return settlement_amount * df

    def forward_rate(
        self,
        fra: ForwardRateAgreement,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate the implied forward rate for the FRA period.

        Args:
            fra: Forward Rate Agreement
            valuation_date: Valuation date

        Returns:
            Forward rate for the accrual period
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        return self._get_forward_rate(fra, valuation_date)

    def par_rate(
        self,
        fra: ForwardRateAgreement,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate the par (fair) FRA rate.

        The par rate is the forward rate — the fixed rate that makes NPV = 0.

        Args:
            fra: Forward Rate Agreement
            valuation_date: Valuation date

        Returns:
            Par FRA rate
        """
        return self.forward_rate(fra, valuation_date)

    def dv01(
        self,
        fra: ForwardRateAgreement,
        valuation_date: Optional[datetime] = None,
        bump_size: float = 0.0001,
    ) -> float:
        """
        Calculate DV01 (dollar value of 1 basis point).

        Uses central difference: DV01 = (P_down - P_up) / (2 * bump)

        Args:
            fra: Forward Rate Agreement
            valuation_date: Valuation date
            bump_size: Bump size (default: 1bp)

        Returns:
            DV01 value
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        original_curve = self.pricing_env.rate_curve
        original_projection = self.projection_curve
        base_rate = original_curve.get_rate(1.0)

        try:
            up_curve = FlatRateCurve(rate=base_rate + bump_size)
            self.pricing_env.rate_curve = up_curve
            self.projection_curve = up_curve
            npv_up = self.price(fra, valuation_date)

            down_curve = FlatRateCurve(rate=base_rate - bump_size)
            self.pricing_env.rate_curve = down_curve
            self.projection_curve = down_curve
            npv_down = self.price(fra, valuation_date)
        finally:
            self.pricing_env.rate_curve = original_curve
            self.projection_curve = original_projection

        return (npv_down - npv_up) / (2 * bump_size)

    def full_analysis(
        self,
        fra: ForwardRateAgreement,
        valuation_date: Optional[datetime] = None,
    ) -> FRAPricingResults:
        """
        Perform full analysis of a FRA.

        Args:
            fra: Forward Rate Agreement
            valuation_date: Valuation date

        Returns:
            FRAPricingResults with all metrics
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        fwd = self._get_forward_rate(fra, valuation_date)
        dcf = fra.day_count_fraction()

        # Settlement calculation
        rate_diff = fwd - fra.fixed_rate
        denominator = 1.0 + fwd * dcf
        settlement_amount = safe_divide(
            fra.notional * dcf * rate_diff, denominator, fallback=0.0
        )

        t_settle = fra.time_to_settlement(valuation_date)
        df = self.pricing_env.get_discount_factor(t_settle)
        settlement_pv = settlement_amount * df

        npv = settlement_pv if not fra.is_expired(valuation_date) else 0.0

        return FRAPricingResults(
            npv=npv,
            forward_rate=fwd,
            settlement_amount=settlement_amount,
            settlement_pv=settlement_pv,
            day_count_fraction=dcf,
            dv01=self.dv01(fra, valuation_date),
            par_rate=fwd,
        )

    def _get_forward_rate(
        self, fra: ForwardRateAgreement, valuation_date: datetime
    ) -> float:
        """
        Calculate the forward rate for the FRA accrual period.

        Uses the projection curve to compute the forward rate between
        the accrual start and accrual end dates.

        Args:
            fra: Forward Rate Agreement
            valuation_date: Valuation date

        Returns:
            Forward rate
        """
        t1 = (fra.accrual_start - valuation_date).days / 365.0
        t2 = (fra.accrual_end - valuation_date).days / 365.0

        if t1 < 0:
            t1 = 0.0

        return self.projection_curve.get_forward_rate(t1, t2)

    def __repr__(self):
        return (
            f"FRAEngine(valuation_date="
            f"{self.pricing_env.valuation_date.date()})"
        )
