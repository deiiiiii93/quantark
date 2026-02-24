"""
Analytical pricing engine for Interest Rate Caps, Floors, and Collars.

Uses Black's model (Black-76) to price each caplet/floorlet individually,
then aggregates to obtain the full cap/floor price.

Black's caplet formula:
    Caplet  = df * dcf * N * [F * N(d1) - K * N(d2)]
    Floorlet = df * dcf * N * [K * N(-d2) - F * N(-d1)]

where:
    d1 = [ln(F/K) + 0.5 * sigma^2 * T_fix] / (sigma * sqrt(T_fix))
    d2 = d1 - sigma * sqrt(T_fix)
    F     = forward rate for the caplet period
    K     = strike rate
    sigma = implied volatility (flat or from vol surface)
    T_fix = time to fixing date (option expiry for this caplet)
    df    = discount factor to payment date
    dcf   = day count fraction for the accrual period
    N     = notional
    N(.)  = standard normal CDF
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from scipy.stats import norm

from asset.rate.product.cap_floor import CapFloor, CapFloorType, Caplet, Collar
from priceenv import PricingEnvironment
from param.rrf import RateCurve, FlatRateCurve
from util.exceptions import ValidationError, MarketDataError
from util.numerical import safe_log, safe_sqrt, safe_divide, is_zero


@dataclass
class CapletPricingResult:
    """
    Pricing result for a single caplet/floorlet.

    Attributes:
        price: Present value of the caplet/floorlet
        forward_rate: Forward rate for this period
        vol: Implied volatility used
        d1: Black's d1
        d2: Black's d2
        intrinsic: Intrinsic value (max(0, F-K)*dcf*N*df for cap)
        time_value: Time value (price - intrinsic)
        delta: Rate delta (sensitivity to forward rate)
        vega: Sensitivity to vol (per 1% vol move)
        accrual_start: Start of accrual period
        accrual_end: End of accrual period
    """

    price: float
    forward_rate: float
    vol: float
    d1: float
    d2: float
    intrinsic: float
    time_value: float
    delta: float
    vega: float
    accrual_start: datetime
    accrual_end: datetime


@dataclass
class CapFloorPricingResults:
    """
    Results container for Cap/Floor pricing.

    Attributes:
        npv: Total present value of the cap/floor
        caplet_prices: Individual caplet/floorlet prices
        caplet_details: Detailed results per caplet
        par_rate: Flat vol implied par strike (forward swap rate)
        flat_vol: Flat vol that reprices the cap/floor (if provided)
        dv01: Dollar value of 1 basis point
        vega: Total vega (sensitivity to 1% vol shift)
    """

    npv: float
    caplet_prices: List[float] = field(default_factory=list)
    caplet_details: List[CapletPricingResult] = field(default_factory=list)
    par_rate: Optional[float] = None
    flat_vol: Optional[float] = None
    dv01: Optional[float] = None
    vega: Optional[float] = None


class CapFloorEngine:
    """
    Analytical pricing engine for Caps, Floors, and Collars.

    Uses Black's model (Black-76) to price each caplet/floorlet. Supports:
    - Flat volatility (same vol for all caplets)
    - Per-caplet volatility from vol surface
    - Single-curve or dual-curve pricing
    """

    def __init__(
        self,
        pricing_env: PricingEnvironment,
        projection_curve: Optional[RateCurve] = None,
        vol: Optional[float] = None,
    ):
        """
        Initialize the Cap/Floor engine.

        Args:
            pricing_env: Pricing environment with discount curve.
                        If vol_surface is set, it provides per-caplet vols.
            projection_curve: Separate curve for forward rate projection.
                             If None, uses the discount curve.
            vol: Flat volatility override. If provided, uses this vol for
                 all caplets instead of the vol surface.
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")
        if pricing_env.rate_curve is None:
            raise MarketDataError("Rate curve is required for cap/floor pricing")

        self.pricing_env = pricing_env
        self.projection_curve = projection_curve or pricing_env.rate_curve
        self.flat_vol = vol

    def price(
        self,
        product: CapFloor,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate the NPV of a Cap or Floor.

        Args:
            product: Cap or Floor to price
            valuation_date: Valuation date (default: pricing env date)

        Returns:
            Net present value (positive for long position)
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if product.is_expired(valuation_date):
            return 0.0

        caplets = product.get_future_caplets(valuation_date)
        total = 0.0

        for caplet in caplets:
            total += self._price_caplet(caplet, product.cap_floor_type, valuation_date)

        return total

    def price_collar(
        self,
        collar: Collar,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate the NPV of a Collar (long cap + short floor).

        Args:
            collar: Collar to price
            valuation_date: Valuation date

        Returns:
            NPV = cap_price - floor_price (borrower's hedge)
        """
        cap_pv = self.price(collar.cap, valuation_date)
        floor_pv = self.price(collar.floor, valuation_date)
        return cap_pv - floor_pv

    def dv01(
        self,
        product: CapFloor,
        valuation_date: Optional[datetime] = None,
        bump_size: float = 0.0001,
    ) -> float:
        """
        Calculate DV01 via central difference.

        Args:
            product: Cap or Floor
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
            npv_up = self.price(product, valuation_date)

            down_curve = FlatRateCurve(rate=base_rate - bump_size)
            self.pricing_env.rate_curve = down_curve
            self.projection_curve = down_curve
            npv_down = self.price(product, valuation_date)
        finally:
            self.pricing_env.rate_curve = original_curve
            self.projection_curve = original_projection

        return (npv_down - npv_up) / (2 * bump_size)

    def vega(
        self,
        product: CapFloor,
        valuation_date: Optional[datetime] = None,
        vol_bump: float = 0.01,
    ) -> float:
        """
        Calculate total vega (sensitivity to flat vol shift of 1%).

        Args:
            product: Cap or Floor
            valuation_date: Valuation date
            vol_bump: Vol bump size (default: 1% = 0.01)

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
            npv_up = self.price(product, valuation_date)

            self.flat_vol = original_vol - vol_bump
            npv_down = self.price(product, valuation_date)
        finally:
            self.flat_vol = original_vol

        return (npv_up - npv_down) / (2 * vol_bump) * 0.01

    def full_analysis(
        self,
        product: CapFloor,
        valuation_date: Optional[datetime] = None,
    ) -> CapFloorPricingResults:
        """
        Perform full analysis of a Cap or Floor.

        Args:
            product: Cap or Floor
            valuation_date: Valuation date

        Returns:
            CapFloorPricingResults with all metrics
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        caplets = product.get_future_caplets(valuation_date)
        caplet_prices = []
        caplet_details = []
        total_npv = 0.0

        for caplet in caplets:
            detail = self._analyze_caplet(
                caplet, product.cap_floor_type, valuation_date
            )
            caplet_details.append(detail)
            caplet_prices.append(detail.price)
            total_npv += detail.price

        total_vega = sum(d.vega for d in caplet_details)
        dv01_val = self.dv01(product, valuation_date)

        return CapFloorPricingResults(
            npv=total_npv,
            caplet_prices=caplet_prices,
            caplet_details=caplet_details,
            flat_vol=self.flat_vol,
            dv01=dv01_val,
            vega=total_vega,
        )

    # =========================================================================
    # Internal Pricing Methods
    # =========================================================================

    def _price_caplet(
        self,
        caplet: Caplet,
        cap_floor_type: CapFloorType,
        valuation_date: datetime,
    ) -> float:
        """
        Price a single caplet/floorlet using Black's formula.

        Args:
            caplet: Caplet to price
            cap_floor_type: CAP or FLOOR
            valuation_date: Valuation date

        Returns:
            Present value of the caplet/floorlet
        """
        # If caplet has already fixed, compute intrinsic value
        if not caplet.is_projected and caplet.index_fixing is not None:
            return self._price_fixed_caplet(caplet, cap_floor_type, valuation_date)

        # Forward rate
        fwd = self._get_forward_rate(caplet, valuation_date)
        K = caplet.strike
        dcf = caplet.day_count_fraction
        N = caplet.notional

        # Time to fixing (option expiry for this caplet)
        t_fix = (caplet.fixing_date - valuation_date).days / 365.0
        if t_fix <= 0:
            t_fix = 0.0

        # Volatility
        sigma = self._get_vol(caplet, fwd, t_fix)

        # Discount factor to payment date
        t_pay = (caplet.payment_date - valuation_date).days / 365.0
        df = self.pricing_env.get_discount_factor(t_pay)

        # Near-expiry or zero vol: use intrinsic
        if is_zero(t_fix) or is_zero(sigma):
            if cap_floor_type == CapFloorType.CAP:
                payoff = max(0.0, fwd - K)
            else:
                payoff = max(0.0, K - fwd)
            return df * dcf * N * payoff

        # Black's formula
        sqrt_t = safe_sqrt(t_fix)
        sigma_sqrt_t = sigma * sqrt_t

        if is_zero(sigma_sqrt_t):
            if cap_floor_type == CapFloorType.CAP:
                payoff = max(0.0, fwd - K)
            else:
                payoff = max(0.0, K - fwd)
            return df * dcf * N * payoff

        d1 = safe_divide(
            safe_log(fwd / K) + 0.5 * sigma * sigma * t_fix,
            sigma_sqrt_t,
            fallback=0.0,
        )
        d2 = d1 - sigma_sqrt_t

        if cap_floor_type == CapFloorType.CAP:
            price = df * dcf * N * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = df * dcf * N * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))

        return max(0.0, price)

    def _price_fixed_caplet(
        self,
        caplet: Caplet,
        cap_floor_type: CapFloorType,
        valuation_date: datetime,
    ) -> float:
        """Price a caplet that has already fixed (intrinsic only)."""
        L = caplet.index_fixing
        K = caplet.strike

        if cap_floor_type == CapFloorType.CAP:
            payoff = max(0.0, L - K)
        else:
            payoff = max(0.0, K - L)

        t_pay = (caplet.payment_date - valuation_date).days / 365.0
        if t_pay <= 0:
            return 0.0

        df = self.pricing_env.get_discount_factor(t_pay)
        return df * caplet.day_count_fraction * caplet.notional * payoff

    def _analyze_caplet(
        self,
        caplet: Caplet,
        cap_floor_type: CapFloorType,
        valuation_date: datetime,
    ) -> CapletPricingResult:
        """Full analysis of a single caplet including Greeks."""
        fwd = self._get_forward_rate(caplet, valuation_date)
        K = caplet.strike
        dcf = caplet.day_count_fraction
        N = caplet.notional

        t_fix = max(0.0, (caplet.fixing_date - valuation_date).days / 365.0)
        sigma = self._get_vol(caplet, fwd, t_fix)

        t_pay = (caplet.payment_date - valuation_date).days / 365.0
        df = self.pricing_env.get_discount_factor(t_pay)

        # Calculate d1, d2
        sqrt_t = safe_sqrt(t_fix)
        sigma_sqrt_t = sigma * sqrt_t if not is_zero(t_fix) else 0.0

        if is_zero(sigma_sqrt_t):
            d1 = 0.0
            d2 = 0.0
        else:
            d1 = safe_divide(
                safe_log(fwd / K) + 0.5 * sigma * sigma * t_fix,
                sigma_sqrt_t,
                fallback=0.0,
            )
            d2 = d1 - sigma_sqrt_t

        # Price
        price = self._price_caplet(caplet, cap_floor_type, valuation_date)

        # Intrinsic
        if cap_floor_type == CapFloorType.CAP:
            intrinsic = df * dcf * N * max(0.0, fwd - K)
        else:
            intrinsic = df * dcf * N * max(0.0, K - fwd)

        # Delta: dPrice/dF
        if cap_floor_type == CapFloorType.CAP:
            delta = df * dcf * N * norm.cdf(d1)
        else:
            delta = -df * dcf * N * norm.cdf(-d1)

        # Vega: dPrice/dSigma (per 1% move)
        vega_val = df * dcf * N * fwd * norm.pdf(d1) * sqrt_t * 0.01

        return CapletPricingResult(
            price=price,
            forward_rate=fwd,
            vol=sigma,
            d1=d1,
            d2=d2,
            intrinsic=intrinsic,
            time_value=price - intrinsic,
            delta=delta,
            vega=vega_val,
            accrual_start=caplet.accrual_start,
            accrual_end=caplet.accrual_end,
        )

    def _get_forward_rate(
        self, caplet: Caplet, valuation_date: datetime
    ) -> float:
        """Get forward rate for a caplet's accrual period."""
        # If already fixed, use the fixing
        if not caplet.is_projected and caplet.index_fixing is not None:
            return caplet.index_fixing

        t1 = (caplet.accrual_start - valuation_date).days / 365.0
        t2 = (caplet.accrual_end - valuation_date).days / 365.0

        if t1 < 0:
            t1 = 0.0

        return self.projection_curve.get_forward_rate(t1, t2)

    def _get_vol(
        self, caplet: Caplet, forward_rate: float, time_to_fixing: float
    ) -> float:
        """
        Get implied volatility for a caplet.

        Priority:
        1. Flat vol override (self.flat_vol)
        2. Vol surface from pricing env (if available)
        3. Raise error if neither available

        Args:
            caplet: Caplet
            forward_rate: Forward rate (used as strike proxy for vol surface)
            time_to_fixing: Time to caplet expiry

        Returns:
            Implied volatility
        """
        if self.flat_vol is not None:
            return self.flat_vol

        if self.pricing_env.vol_surface is not None:
            # Use caplet strike as lookup key
            # For rate vol surfaces, strike = rate level
            return self.pricing_env.vol_surface.get_vol(
                caplet.strike, time_to_fixing, forward_rate
            )

        raise MarketDataError(
            "No volatility provided: set flat_vol or provide vol_surface "
            "in PricingEnvironment"
        )

    def __repr__(self):
        vol_str = f", flat_vol={self.flat_vol:.2%}" if self.flat_vol else ""
        return (
            f"CapFloorEngine(valuation_date="
            f"{self.pricing_env.valuation_date.date()}{vol_str})"
        )
