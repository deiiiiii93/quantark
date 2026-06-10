"""
Analytical pricing engine for European Swaptions.

Uses Black's model (Black-76) for European swaption pricing:

Payer swaption:
    V = A * [S * N(d1) - K * N(d2)]

Receiver swaption:
    V = A * [K * N(-d2) - S * N(-d1)]

where:
    d1 = [ln(S/K) + 0.5 * sigma^2 * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    S     = forward swap rate
    K     = strike (fixed rate of underlying swap)
    sigma = swaption implied volatility
    T     = time to exercise
    A     = annuity of the underlying swap (PV of 1bp on fixed leg)
    N(.)  = standard normal CDF

Alternatively, supports Bachelier (normal) model:
    Payer = A * [(S-K) * N(d) + sigma * sqrt(T) * n(d)]
    where d = (S - K) / (sigma * sqrt(T))
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from scipy.stats import norm

from dateutil.relativedelta import relativedelta

from quantark.asset.rate.product.swaption import Swaption, SwaptionType
from quantark.priceenv import PricingEnvironment
from quantark.param.rrf import RateCurve, FlatRateCurve
from quantark.util.calendar import calculate_day_count_fraction
from quantark.util.exceptions import ValidationError, MarketDataError
from quantark.util.numerical import safe_log, safe_sqrt, safe_divide, is_zero


class SwaptionModelType(Enum):
    """Model type for swaption pricing."""

    BLACK = "black"  # Black-76 (lognormal)
    BACHELIER = "bachelier"  # Normal (Bachelier)


@dataclass
class SwaptionPricingResults:
    """
    Results container for swaption pricing.

    Attributes:
        npv: Net present value of the swaption
        forward_swap_rate: Forward swap rate (par rate of underlying)
        annuity: Annuity (PV01) of the underlying swap fixed leg
        d1: Black's d1 (or Bachelier's d)
        d2: Black's d2
        implied_vol: Volatility used for pricing
        model: Model type used (BLACK or BACHELIER)
        intrinsic: Intrinsic value of the swaption
        time_value: Time value (npv - intrinsic)
        delta: Sensitivity to forward swap rate
        vega: Sensitivity to vol (per 1% vol move)
        dv01: Dollar value of 1 basis point
    """

    npv: float
    forward_swap_rate: float
    annuity: float
    d1: float
    d2: float
    implied_vol: float
    model: SwaptionModelType = SwaptionModelType.BLACK
    intrinsic: Optional[float] = None
    time_value: Optional[float] = None
    delta: Optional[float] = None
    vega: Optional[float] = None
    dv01: Optional[float] = None


class SwaptionEngine:
    """
    Analytical pricing engine for European Swaptions.

    Supports Black's model (lognormal) and Bachelier model (normal)
    for European swaptions. Uses the forward swap rate and annuity
    computed from the rate curve.
    """

    def __init__(
        self,
        pricing_env: PricingEnvironment,
        projection_curve: Optional[RateCurve] = None,
        vol: Optional[float] = None,
        model: SwaptionModelType = SwaptionModelType.BLACK,
    ):
        """
        Initialize the swaption engine.

        Args:
            pricing_env: Pricing environment with discount curve
            projection_curve: Separate curve for forward rate projection.
                             If None, uses discount curve.
            vol: Flat volatility. If None, uses vol surface from pricing env.
            model: Pricing model (BLACK or BACHELIER)
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")
        if pricing_env.rate_curve is None:
            raise MarketDataError("Rate curve is required for swaption pricing")

        self.pricing_env = pricing_env
        self.projection_curve = projection_curve or pricing_env.rate_curve
        self.flat_vol = vol
        self.model = model

    def price(
        self,
        swaption: Swaption,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate the NPV of a European swaption.

        Args:
            swaption: Swaption to price
            valuation_date: Valuation date (default: pricing env date)

        Returns:
            Net present value (always non-negative for long position)
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if swaption.is_expired(valuation_date):
            return 0.0

        # Forward swap rate and annuity
        S = self._forward_swap_rate(swaption, valuation_date)
        A = self._annuity(swaption, valuation_date)
        K = swaption.fixed_rate
        T = swaption.time_to_expiry(valuation_date)
        sigma = self._get_vol(swaption, S, T)

        if self.model == SwaptionModelType.BLACK:
            return self._black_price(S, K, T, sigma, A, swaption.swaption_type)
        else:
            return self._bachelier_price(S, K, T, sigma, A, swaption.swaption_type)

    def forward_swap_rate(
        self,
        swaption: Swaption,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate the forward swap rate for the underlying swap.

        The forward swap rate is the fixed rate that makes the forward
        value of the underlying swap equal to zero.

        Args:
            swaption: Swaption
            valuation_date: Valuation date

        Returns:
            Forward swap rate
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        return self._forward_swap_rate(swaption, valuation_date)

    def annuity(
        self,
        swaption: Swaption,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate the annuity (PV01) of the underlying swap.

        The annuity is the sum of discounted day count fractions
        weighted by notional: A = sum(df_i * dcf_i * N_i)

        Args:
            swaption: Swaption
            valuation_date: Valuation date

        Returns:
            Annuity value
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        return self._annuity(swaption, valuation_date)

    def dv01(
        self,
        swaption: Swaption,
        valuation_date: Optional[datetime] = None,
        bump_size: float = 0.0001,
    ) -> float:
        """
        Calculate DV01 via central difference.

        Args:
            swaption: Swaption
            valuation_date: Valuation date
            bump_size: Rate bump (default: 1bp)

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
            npv_up = self.price(swaption, valuation_date)

            down_curve = FlatRateCurve(rate=base_rate - bump_size)
            self.pricing_env.rate_curve = down_curve
            self.projection_curve = down_curve
            npv_down = self.price(swaption, valuation_date)
        finally:
            self.pricing_env.rate_curve = original_curve
            self.projection_curve = original_projection

        return (npv_down - npv_up) / (2 * bump_size)

    def vega(
        self,
        swaption: Swaption,
        valuation_date: Optional[datetime] = None,
        vol_bump: float = 0.01,
    ) -> float:
        """
        Calculate vega (sensitivity to 1% vol shift).

        Args:
            swaption: Swaption
            valuation_date: Valuation date
            vol_bump: Vol bump size (default: 1%)

        Returns:
            Vega (dollar change per 1% vol increase)
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if self.flat_vol is None:
            return 0.0

        original_vol = self.flat_vol

        try:
            self.flat_vol = original_vol + vol_bump
            npv_up = self.price(swaption, valuation_date)

            self.flat_vol = original_vol - vol_bump
            npv_down = self.price(swaption, valuation_date)
        finally:
            self.flat_vol = original_vol

        return (npv_up - npv_down) / (2 * vol_bump) * 0.01

    def full_analysis(
        self,
        swaption: Swaption,
        valuation_date: Optional[datetime] = None,
    ) -> SwaptionPricingResults:
        """
        Perform full analysis of a swaption.

        Args:
            swaption: Swaption
            valuation_date: Valuation date

        Returns:
            SwaptionPricingResults with all metrics
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        S = self._forward_swap_rate(swaption, valuation_date)
        A = self._annuity(swaption, valuation_date)
        K = swaption.fixed_rate
        T = swaption.time_to_expiry(valuation_date)
        sigma = self._get_vol(swaption, S, T)

        # Calculate d1, d2
        d1, d2 = self._calc_d1_d2(S, K, T, sigma)

        # Price
        npv = self.price(swaption, valuation_date)

        # Intrinsic (exercise value assuming immediate exercise)
        if swaption.swaption_type == SwaptionType.PAYER:
            intrinsic = max(0.0, A * (S - K))
        else:
            intrinsic = max(0.0, A * (K - S))

        # Delta: dV/dS = A * N(d1) for payer, -A * N(-d1) for receiver
        if self.model == SwaptionModelType.BLACK:
            if swaption.swaption_type == SwaptionType.PAYER:
                delta = A * norm.cdf(d1)
            else:
                delta = -A * norm.cdf(-d1)
        else:
            # Bachelier delta
            if not is_zero(T) and not is_zero(sigma):
                d = safe_divide(S - K, sigma * safe_sqrt(T), fallback=0.0)
                if swaption.swaption_type == SwaptionType.PAYER:
                    delta = A * norm.cdf(d)
                else:
                    delta = -A * norm.cdf(-d)
            else:
                delta = A if S > K else 0.0

        # Vega
        vega_val = self.vega(swaption, valuation_date)

        # DV01
        dv01_val = self.dv01(swaption, valuation_date)

        return SwaptionPricingResults(
            npv=npv,
            forward_swap_rate=S,
            annuity=A,
            d1=d1,
            d2=d2,
            implied_vol=sigma,
            model=self.model,
            intrinsic=intrinsic,
            time_value=npv - intrinsic,
            delta=delta,
            vega=vega_val,
            dv01=dv01_val,
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _black_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        A: float,
        swaption_type: SwaptionType,
    ) -> float:
        """
        Black's formula for swaption pricing.

        Args:
            S: Forward swap rate
            K: Strike rate
            T: Time to expiry
            sigma: Implied volatility (lognormal)
            A: Annuity
            swaption_type: PAYER or RECEIVER

        Returns:
            Swaption price
        """
        if is_zero(T) or is_zero(sigma):
            # At expiry or zero vol: intrinsic value
            if swaption_type == SwaptionType.PAYER:
                return max(0.0, A * (S - K))
            else:
                return max(0.0, A * (K - S))

        d1, d2 = self._calc_d1_d2(S, K, T, sigma)

        if swaption_type == SwaptionType.PAYER:
            price = A * (S * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = A * (K * norm.cdf(-d2) - S * norm.cdf(-d1))

        return max(0.0, price)

    def _bachelier_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        A: float,
        swaption_type: SwaptionType,
    ) -> float:
        """
        Bachelier (normal) model for swaption pricing.

        Args:
            S: Forward swap rate
            K: Strike rate
            T: Time to expiry
            sigma: Implied volatility (normal, in rate units)
            A: Annuity
            swaption_type: PAYER or RECEIVER

        Returns:
            Swaption price
        """
        if is_zero(T) or is_zero(sigma):
            if swaption_type == SwaptionType.PAYER:
                return max(0.0, A * (S - K))
            else:
                return max(0.0, A * (K - S))

        sqrt_t = safe_sqrt(T)
        sigma_sqrt_t = sigma * sqrt_t

        d = safe_divide(S - K, sigma_sqrt_t, fallback=0.0)

        if swaption_type == SwaptionType.PAYER:
            price = A * ((S - K) * norm.cdf(d) + sigma_sqrt_t * norm.pdf(d))
        else:
            price = A * ((K - S) * norm.cdf(-d) + sigma_sqrt_t * norm.pdf(d))

        return max(0.0, price)

    def _calc_d1_d2(
        self, S: float, K: float, T: float, sigma: float
    ) -> tuple:
        """Calculate Black's d1 and d2."""
        if is_zero(T) or is_zero(sigma) or is_zero(K) or is_zero(S):
            return 0.0, 0.0

        sqrt_t = safe_sqrt(T)
        sigma_sqrt_t = sigma * sqrt_t

        if is_zero(sigma_sqrt_t):
            return 0.0, 0.0

        d1 = safe_divide(
            safe_log(S / K) + 0.5 * sigma * sigma * T,
            sigma_sqrt_t,
            fallback=0.0,
        )
        d2 = d1 - sigma_sqrt_t

        return d1, d2

    def _forward_swap_rate(
        self, swaption: Swaption, valuation_date: datetime
    ) -> float:
        """
        Calculate the forward swap rate.

        The forward swap rate is the par rate of the underlying swap,
        computed as:
            S = sum(df_i * fwd_i * dcf_i * N_i) / sum(df_i * dcf_i * N_i)
              = PV(floating leg) / Annuity

        For simplicity with flat/smooth curves, we compute the forward
        rate from the discount factors at swap start and end:
            S = (df(T_start) - df(T_end)) / Annuity

        Args:
            swaption: Swaption
            valuation_date: Valuation date

        Returns:
            Forward swap rate
        """
        t_start = (swaption.swap_start_date - valuation_date).days / 365.0
        t_end = (swaption.swap_end_date - valuation_date).days / 365.0

        if t_start < 0:
            t_start = 0.0

        df_start = self.projection_curve.get_discount_factor(t_start)
        df_end = self.projection_curve.get_discount_factor(t_end)

        annuity = self._annuity(swaption, valuation_date)

        if is_zero(annuity):
            # Fallback: use simple forward rate
            return self.projection_curve.get_forward_rate(t_start, t_end)

        # Numerator must include notional to match annuity (which is notional-weighted)
        return safe_divide(
            swaption.notional * (df_start - df_end), annuity, fallback=0.0
        )

    def _annuity(
        self, swaption: Swaption, valuation_date: datetime
    ) -> float:
        """
        Calculate the annuity of the underlying swap.

        Annuity = sum(df(t_i) * dcf_i * N_i) for each fixed leg period.

        We generate the fixed leg schedule from the swap parameters
        and compute the discounted day count fractions.

        Args:
            swaption: Swaption
            valuation_date: Valuation date

        Returns:
            Annuity value
        """
        months_per_period = 12 // swaption.payment_frequency.periods_per_year
        notional = swaption.notional

        # Generate fixed leg dates (backward from swap end)
        period_ends = []
        current = swaption.swap_end_date
        while current > swaption.swap_start_date:
            period_ends.append(current)
            current = current - relativedelta(months=months_per_period)
        period_ends.reverse()

        annuity = 0.0
        accrual_start = swaption.swap_start_date

        for period_end in period_ends:
            dcf = calculate_day_count_fraction(
                accrual_start, period_end, swaption.fixed_day_count
            )
            t_pay = (period_end - valuation_date).days / 365.0

            # Get period notional (for amortizing)
            if swaption.notional_schedule is not None:
                period_notional = swaption.notional_schedule.get_notional(accrual_start)
            else:
                period_notional = notional

            df = self.pricing_env.get_discount_factor(t_pay)
            annuity += df * dcf * period_notional

            accrual_start = period_end

        return annuity

    def _get_vol(
        self, swaption: Swaption, forward_rate: float, time_to_expiry: float
    ) -> float:
        """
        Get swaption implied volatility.

        Priority:
        1. Flat vol override
        2. Vol surface from pricing env
        3. Error

        Args:
            swaption: Swaption
            forward_rate: Forward swap rate
            time_to_expiry: Time to exercise

        Returns:
            Implied volatility
        """
        if self.flat_vol is not None:
            return self.flat_vol

        if self.pricing_env.vol_surface is not None:
            return self.pricing_env.vol_surface.get_vol(
                swaption.fixed_rate, time_to_expiry, forward_rate
            )

        raise MarketDataError(
            "No volatility provided: set flat_vol or provide vol_surface "
            "in PricingEnvironment"
        )

    def __repr__(self):
        vol_str = f", vol={self.flat_vol:.2%}" if self.flat_vol else ""
        return (
            f"SwaptionEngine({self.model.value}, "
            f"valuation_date={self.pricing_env.valuation_date.date()}"
            f"{vol_str})"
        )
