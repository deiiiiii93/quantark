"""
Discount-based pricing engine for Interest Rate Swaps.

Provides pricing, par rate calculations, and risk metrics for IRS and Basis Swaps.
Supports both single-curve and dual-curve pricing methodologies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Union
import math

from quantark.asset.rate.product.irs import (
    InterestRateSwap,
    BasisSwap,
    FixedLeg,
    FloatingLeg,
    SwapLeg,
    SwapDirection,
)
from quantark.asset.bond.schedule.cashflow import CashFlow, FloatingCashFlow
from quantark.priceenv import PricingEnvironment
from quantark.param.rrf import RateCurve, FlatRateCurve
from quantark.util.exceptions import ValidationError, MarketDataError


@dataclass
class IRSPricingResults:
    """
    Results container for IRS pricing.

    Attributes:
        npv: Net Present Value (receive leg PV - pay leg PV)
        receive_leg_pv: Present value of receive leg
        pay_leg_pv: Present value of pay leg
        par_rate: Fair fixed rate for at-market swap
        fixed_leg_accrued: Accrued interest on fixed leg
        floating_leg_accrued: Accrued interest on floating leg
        net_accrued: Net accrued interest
        dv01: Dollar value of 1 basis point (parallel shift sensitivity)
        bpv: Basis point value (same as DV01)
        duration: Modified duration
        convexity: Convexity measure
        fixed_leg_duration: Duration of fixed leg
        floating_leg_duration: Duration of floating leg
        key_rate_durations: Bucket sensitivities (optional)
        weighted_average_life: WAL in years
    """

    npv: float
    receive_leg_pv: float
    pay_leg_pv: float
    par_rate: Optional[float] = None
    fixed_leg_accrued: Optional[float] = None
    floating_leg_accrued: Optional[float] = None
    net_accrued: Optional[float] = None
    dv01: Optional[float] = None
    bpv: Optional[float] = None
    duration: Optional[float] = None
    convexity: Optional[float] = None
    fixed_leg_duration: Optional[float] = None
    floating_leg_duration: Optional[float] = None
    key_rate_durations: Optional[Dict[float, float]] = None
    weighted_average_life: Optional[float] = None


@dataclass
class BasisSwapPricingResults:
    """
    Results container for Basis Swap pricing.

    Attributes:
        npv: Net Present Value (leg2 PV - leg1 PV)
        leg1_pv: Present value of leg 1 (pay leg)
        leg2_pv: Present value of leg 2 (receive leg)
        par_spread: Fair spread for at-market basis swap
        dv01: Dollar value of 1 basis point
        duration: Modified duration
    """

    npv: float
    leg1_pv: float
    leg2_pv: float
    par_spread: Optional[float] = None
    dv01: Optional[float] = None
    duration: Optional[float] = None


class IRSDiscountEngine:
    """
    Pricing engine for Interest Rate Swaps using discounted cashflows.

    Supports:
    - Single-curve pricing (same curve for discounting and projection)
    - NPV calculation for vanilla IRS and basis swaps
    - Par rate / par spread calculation
    - Risk metrics (DV01, duration, key rate durations)
    - Accrued interest calculations

    The engine uses the rate curve from PricingEnvironment for both
    discounting cashflows and projecting forward rates.
    """

    def __init__(
        self,
        pricing_env: PricingEnvironment,
        projection_curve: Optional[RateCurve] = None,
    ):
        """
        Initialize IRS discount engine.

        Args:
            pricing_env: Pricing environment with discount curve
            projection_curve: Optional separate curve for forward rate projection
                             (if None, uses discount curve from pricing_env)
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")

        if pricing_env.rate_curve is None:
            raise MarketDataError("Rate curve is required for IRS pricing")

        self.pricing_env = pricing_env
        self.projection_curve = projection_curve or pricing_env.rate_curve

    # =========================================================================
    # Core Pricing Methods
    # =========================================================================

    def price(
        self,
        swap: Union[InterestRateSwap, BasisSwap],
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate swap NPV (Net Present Value).

        For InterestRateSwap:
            NPV = PV(receive_leg) - PV(pay_leg)

        For BasisSwap:
            NPV = PV(leg2) - PV(leg1)

        Args:
            swap: Interest rate swap or basis swap to price
            valuation_date: Date to value the swap (default: pricing env date)

        Returns:
            Net present value of the swap
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if swap.is_expired(valuation_date):
            return 0.0

        if isinstance(swap, InterestRateSwap):
            # Update forward rates on floating leg
            swap.update_forward_rates(self.projection_curve, valuation_date)

            receive_pv = self._leg_pv(swap.receive_leg, valuation_date)
            pay_pv = self._leg_pv(swap.pay_leg, valuation_date)
            return receive_pv - pay_pv

        elif isinstance(swap, BasisSwap):
            # Update forward rates on both legs
            swap.update_forward_rates(self.projection_curve, valuation_date)

            leg2_pv = self._leg_pv(swap.leg2, valuation_date)
            leg1_pv = self._leg_pv(swap.leg1, valuation_date)
            return leg2_pv - leg1_pv

        else:
            raise ValidationError(f"Unknown swap type: {type(swap)}")

    def npv(
        self,
        swap: Union[InterestRateSwap, BasisSwap],
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate swap NPV (alias for price method).

        Args:
            swap: Swap to price
            valuation_date: Valuation date

        Returns:
            Net present value
        """
        return self.price(swap, valuation_date)

    def _leg_pv(
        self,
        leg: SwapLeg,
        valuation_date: datetime,
    ) -> float:
        """
        Calculate present value of a single leg.

        Args:
            leg: Swap leg to price
            valuation_date: Valuation date

        Returns:
            Present value of the leg
        """
        cashflows = leg.get_cashflows(valuation_date)

        if not cashflows:
            return 0.0

        pv = 0.0
        for cf in cashflows:
            time_to_payment = (cf.payment_date - valuation_date).days / 365.0

            if time_to_payment < 0:
                continue

            df = self.pricing_env.get_discount_factor(time_to_payment)
            pv += cf.amount * df

        return pv

    def fixed_leg_pv(
        self,
        leg: FixedLeg,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate present value of a fixed leg.

        Args:
            leg: Fixed leg to price
            valuation_date: Valuation date

        Returns:
            Present value of fixed leg cashflows
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        return self._leg_pv(leg, valuation_date)

    def floating_leg_pv(
        self,
        leg: FloatingLeg,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate present value of a floating leg.

        Args:
            leg: Floating leg to price
            valuation_date: Valuation date

        Returns:
            Present value of floating leg cashflows
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        # Update forward rates
        leg.update_forward_rates(self.projection_curve, valuation_date)

        return self._leg_pv(leg, valuation_date)

    # =========================================================================
    # Par Rate and Par Spread Calculations
    # =========================================================================

    def par_rate(
        self,
        swap: InterestRateSwap,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate the par (fair) fixed rate for a swap.

        The par rate is the fixed rate that makes the swap NPV equal to zero.

        Par Rate = PV(floating leg) / Annuity

        where Annuity = sum of discounted day count fractions weighted by notional.

        Args:
            swap: Interest rate swap
            valuation_date: Valuation date

        Returns:
            Par fixed rate
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        # Update forward rates
        swap.update_forward_rates(self.projection_curve, valuation_date)

        # Calculate PV of floating leg
        floating_pv = self.floating_leg_pv(swap.floating_leg, valuation_date)

        # Calculate fixed leg annuity (sum of df * dcf * notional)
        annuity = self._calculate_annuity(swap.fixed_leg, valuation_date)

        if abs(annuity) < 1e-10:
            raise ValidationError("Annuity is too small to calculate par rate")

        return floating_pv / annuity

    def par_spread(
        self,
        swap: Union[InterestRateSwap, BasisSwap],
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate the par spread for a swap.

        For InterestRateSwap: The spread that makes floating leg equal to fixed leg PV.
        For BasisSwap: The spread on leg2 that makes NPV equal to zero.

        Args:
            swap: Swap to calculate par spread for
            valuation_date: Valuation date

        Returns:
            Par spread
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if isinstance(swap, BasisSwap):
            # For basis swap: find spread on leg2 to zero NPV
            swap.update_forward_rates(self.projection_curve, valuation_date)

            # PV of leg1 (without spread adjustment)
            leg1_pv = self.floating_leg_pv(swap.leg1, valuation_date)

            # Calculate leg2 annuity
            annuity2 = self._calculate_floating_annuity(swap.leg2, valuation_date)

            # PV of leg2 without spread
            original_spread = swap.leg2.spread
            swap.leg2.spread = 0.0
            swap.leg2._cached_schedule = swap.leg2._generate_schedule()
            swap.leg2.update_forward_rates(self.projection_curve, valuation_date)
            leg2_base_pv = self.floating_leg_pv(swap.leg2, valuation_date)

            # Restore original spread
            swap.leg2.spread = original_spread
            swap.leg2._cached_schedule = swap.leg2._generate_schedule()

            if abs(annuity2) < 1e-10:
                raise ValidationError("Annuity is too small to calculate par spread")

            # Spread needed on leg2 to make NPV = 0
            # leg2_base_pv + spread * annuity2 = leg1_pv
            return (leg1_pv - leg2_base_pv) / annuity2 + swap.leg1.spread

        else:
            # For IRS: spread on floating leg to make it equal to fixed leg PV
            swap.update_forward_rates(self.projection_curve, valuation_date)

            fixed_pv = self.fixed_leg_pv(swap.fixed_leg, valuation_date)
            floating_annuity = self._calculate_floating_annuity(
                swap.floating_leg, valuation_date
            )

            # PV of floating without spread
            original_spread = swap.floating_leg.spread
            swap.floating_leg.spread = 0.0
            swap.floating_leg._cached_schedule = swap.floating_leg._generate_schedule()
            swap.floating_leg.update_forward_rates(
                self.projection_curve, valuation_date
            )
            floating_base_pv = self.floating_leg_pv(swap.floating_leg, valuation_date)

            # Restore
            swap.floating_leg.spread = original_spread
            swap.floating_leg._cached_schedule = swap.floating_leg._generate_schedule()

            if abs(floating_annuity) < 1e-10:
                raise ValidationError("Annuity is too small to calculate par spread")

            return (fixed_pv - floating_base_pv) / floating_annuity

    def _calculate_annuity(
        self,
        leg: FixedLeg,
        valuation_date: datetime,
    ) -> float:
        """
        Calculate the annuity (PV01) of a fixed leg.

        Annuity = sum(df_i * dcf_i * notional_i) for all future cashflows

        Args:
            leg: Fixed leg
            valuation_date: Valuation date

        Returns:
            Annuity value
        """
        cashflows = leg.get_fixed_cashflows()

        annuity = 0.0
        for cf in cashflows:
            if cf.payment_date <= valuation_date:
                continue

            time_to_payment = (cf.payment_date - valuation_date).days / 365.0
            df = self.pricing_env.get_discount_factor(time_to_payment)
            annuity += df * cf.day_count_fraction * cf.notional

        return annuity

    def _calculate_floating_annuity(
        self,
        leg: FloatingLeg,
        valuation_date: datetime,
    ) -> float:
        """
        Calculate the annuity for a floating leg.

        Args:
            leg: Floating leg
            valuation_date: Valuation date

        Returns:
            Annuity value
        """
        cashflows = leg.get_floating_cashflows()

        annuity = 0.0
        for cf in cashflows:
            if cf.payment_date <= valuation_date:
                continue

            time_to_payment = (cf.payment_date - valuation_date).days / 365.0
            df = self.pricing_env.get_discount_factor(time_to_payment)
            annuity += df * cf.day_count_fraction * cf.notional

        return annuity

    # =========================================================================
    # Accrued Interest
    # =========================================================================

    def accrued_interest(
        self,
        swap: Union[InterestRateSwap, BasisSwap],
        settlement_date: Optional[datetime] = None,
    ) -> Tuple[float, float]:
        """
        Calculate accrued interest for both legs.

        Args:
            swap: Swap to calculate accrued for
            settlement_date: Settlement date

        Returns:
            Tuple of (pay_leg_accrued, receive_leg_accrued)
        """
        if settlement_date is None:
            settlement_date = self.pricing_env.valuation_date

        if isinstance(swap, InterestRateSwap):
            pay_accrued = swap.pay_leg.calculate_accrued_interest(settlement_date)
            receive_accrued = swap.receive_leg.calculate_accrued_interest(
                settlement_date
            )
        else:
            pay_accrued = swap.leg1.calculate_accrued_interest(settlement_date)
            receive_accrued = swap.leg2.calculate_accrued_interest(settlement_date)

        return (pay_accrued, receive_accrued)

    def net_accrued_interest(
        self,
        swap: Union[InterestRateSwap, BasisSwap],
        settlement_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate net accrued interest (receive - pay).

        Args:
            swap: Swap to calculate accrued for
            settlement_date: Settlement date

        Returns:
            Net accrued interest
        """
        pay_accrued, receive_accrued = self.accrued_interest(swap, settlement_date)
        return receive_accrued - pay_accrued

    # =========================================================================
    # Risk Metrics
    # =========================================================================

    def dv01(
        self,
        swap: Union[InterestRateSwap, BasisSwap],
        valuation_date: Optional[datetime] = None,
        bump_size: float = 0.0001,
    ) -> float:
        """
        Calculate DV01 (dollar value of 1 basis point).

        DV01 measures the change in swap value for a 1bp parallel shift
        in the interest rate curve.

        Args:
            swap: Swap to calculate DV01 for
            valuation_date: Valuation date
            bump_size: Size of rate bump (default: 1bp = 0.0001)

        Returns:
            DV01 value
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        # Base NPV
        base_npv = self.price(swap, valuation_date)

        # Create bumped curve
        original_curve = self.pricing_env.rate_curve
        base_rate = original_curve.get_rate(1.0)  # Use 1Y rate as reference

        up_curve = FlatRateCurve(rate=base_rate + bump_size)
        down_curve = FlatRateCurve(rate=base_rate - bump_size)

        # Price with up curve
        self.pricing_env.rate_curve = up_curve
        self.projection_curve = up_curve
        npv_up = self.price(swap, valuation_date)

        # Price with down curve
        self.pricing_env.rate_curve = down_curve
        self.projection_curve = down_curve
        npv_down = self.price(swap, valuation_date)

        # Restore original curve
        self.pricing_env.rate_curve = original_curve
        self.projection_curve = original_curve

        # DV01 = (P_down - P_up) / (2 * bump)
        # Note: swap value increases when rates decrease
        dv01 = (npv_down - npv_up) / (2 * bump_size)

        return dv01

    def bpv(
        self,
        swap: Union[InterestRateSwap, BasisSwap],
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate BPV (Basis Point Value) - alias for DV01.

        Args:
            swap: Swap to calculate BPV for
            valuation_date: Valuation date

        Returns:
            BPV value
        """
        return self.dv01(swap, valuation_date)

    def duration(
        self,
        swap: Union[InterestRateSwap, BasisSwap],
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate modified duration.

        Duration = -dP/dY * (1/P) = DV01 / (P * 0.0001)

        For swaps, this is the percentage change per 1bp rate move.

        Args:
            swap: Swap to calculate duration for
            valuation_date: Valuation date

        Returns:
            Modified duration
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        npv = abs(self.price(swap, valuation_date))

        if npv < 1e-10:
            return 0.0

        dv01 = self.dv01(swap, valuation_date)

        # Duration in years: dv01 / (npv * 0.0001)
        return dv01 / (npv * 0.0001)

    def convexity(
        self,
        swap: Union[InterestRateSwap, BasisSwap],
        valuation_date: Optional[datetime] = None,
        bump_size: float = 0.0001,
    ) -> float:
        """
        Calculate convexity.

        Convexity measures the curvature of the price-yield relationship.

        Args:
            swap: Swap to calculate convexity for
            valuation_date: Valuation date
            bump_size: Size of rate bump

        Returns:
            Convexity measure
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        # Get prices at different rate levels
        original_curve = self.pricing_env.rate_curve
        base_rate = original_curve.get_rate(1.0)

        # Base price
        base_npv = self.price(swap, valuation_date)

        if abs(base_npv) < 1e-10:
            return 0.0

        # Up price
        up_curve = FlatRateCurve(rate=base_rate + bump_size)
        self.pricing_env.rate_curve = up_curve
        self.projection_curve = up_curve
        npv_up = self.price(swap, valuation_date)

        # Down price
        down_curve = FlatRateCurve(rate=base_rate - bump_size)
        self.pricing_env.rate_curve = down_curve
        self.projection_curve = down_curve
        npv_down = self.price(swap, valuation_date)

        # Restore
        self.pricing_env.rate_curve = original_curve
        self.projection_curve = original_curve

        # Convexity = (P_up + P_down - 2*P_base) / (P_base * dy^2)
        convexity = (npv_up + npv_down - 2 * base_npv) / (
            base_npv * bump_size * bump_size
        )

        return convexity

    def key_rate_durations(
        self,
        swap: Union[InterestRateSwap, BasisSwap],
        valuation_date: Optional[datetime] = None,
        key_tenors: Optional[List[float]] = None,
        bump_size: float = 0.0001,
    ) -> Dict[float, float]:
        """
        Calculate key rate durations (bucket sensitivities).

        Key rate durations measure the sensitivity to specific points
        on the yield curve.

        Args:
            swap: Swap to calculate KRDs for
            valuation_date: Valuation date
            key_tenors: List of tenors (in years) to calculate KRDs for
                       Default: [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
            bump_size: Size of rate bump

        Returns:
            Dictionary mapping tenor to KRD value
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if key_tenors is None:
            key_tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

        # For a flat curve implementation, KRDs are approximate
        # A full implementation would use a bootstrapped curve

        npv = abs(self.price(swap, valuation_date))
        if npv < 1e-10:
            return {tenor: 0.0 for tenor in key_tenors}

        krds = {}
        original_curve = self.pricing_env.rate_curve

        for tenor in key_tenors:
            base_rate = original_curve.get_rate(tenor)

            # Simple approximation: bump all rates
            # A proper implementation would bump only the specific tenor
            up_curve = FlatRateCurve(rate=base_rate + bump_size)
            self.pricing_env.rate_curve = up_curve
            self.projection_curve = up_curve
            npv_up = self.price(swap, valuation_date)

            down_curve = FlatRateCurve(rate=base_rate - bump_size)
            self.pricing_env.rate_curve = down_curve
            self.projection_curve = down_curve
            npv_down = self.price(swap, valuation_date)

            krd = (npv_down - npv_up) / (2 * bump_size * npv)
            krds[tenor] = krd

        # Restore
        self.pricing_env.rate_curve = original_curve
        self.projection_curve = original_curve

        return krds

    def weighted_average_life(
        self,
        swap: InterestRateSwap,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate Weighted Average Life (WAL).

        WAL = sum(t_i * notional_i) / sum(notional_i)

        Args:
            swap: Swap to calculate WAL for
            valuation_date: Valuation date

        Returns:
            Weighted average life in years
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        cashflows = swap.fixed_leg.get_fixed_cashflows()

        weighted_time = 0.0
        total_notional = 0.0

        for cf in cashflows:
            if cf.payment_date <= valuation_date:
                continue

            time_to_payment = (cf.payment_date - valuation_date).days / 365.0
            weighted_time += time_to_payment * cf.notional
            total_notional += cf.notional

        if total_notional < 1e-10:
            return 0.0

        return weighted_time / total_notional

    # =========================================================================
    # Full Analysis
    # =========================================================================

    def full_analysis(
        self,
        swap: InterestRateSwap,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
    ) -> IRSPricingResults:
        """
        Perform full analysis of an IRS.

        Args:
            swap: Interest rate swap to analyze
            valuation_date: Valuation date
            settlement_date: Settlement date for accrued calculation

        Returns:
            IRSPricingResults with all metrics
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if settlement_date is None:
            settlement_date = valuation_date

        # Update forward rates
        swap.update_forward_rates(self.projection_curve, valuation_date)

        # Calculate leg PVs
        receive_pv = self._leg_pv(swap.receive_leg, valuation_date)
        pay_pv = self._leg_pv(swap.pay_leg, valuation_date)
        npv = receive_pv - pay_pv

        # Par rate
        try:
            par = self.par_rate(swap, valuation_date)
        except ValidationError:
            par = None

        # Accrued interest
        pay_accrued, receive_accrued = self.accrued_interest(swap, settlement_date)

        if swap.direction == SwapDirection.PAYER:
            fixed_accrued = pay_accrued
            float_accrued = receive_accrued
        else:
            fixed_accrued = receive_accrued
            float_accrued = pay_accrued

        net_accrued = receive_accrued - pay_accrued

        # Risk metrics
        dv01 = self.dv01(swap, valuation_date)
        dur = self.duration(swap, valuation_date)
        conv = self.convexity(swap, valuation_date)
        wal = self.weighted_average_life(swap, valuation_date)

        # Leg durations (simplified)
        fixed_dur = self._calculate_annuity(swap.fixed_leg, valuation_date)
        float_annuity = self._calculate_floating_annuity(
            swap.floating_leg, valuation_date
        )

        fixed_pv = self.fixed_leg_pv(swap.fixed_leg, valuation_date)
        float_pv = self.floating_leg_pv(swap.floating_leg, valuation_date)

        fixed_leg_dur = fixed_dur / fixed_pv if abs(fixed_pv) > 1e-10 else 0
        float_leg_dur = float_annuity / float_pv if abs(float_pv) > 1e-10 else 0

        return IRSPricingResults(
            npv=npv,
            receive_leg_pv=receive_pv,
            pay_leg_pv=pay_pv,
            par_rate=par,
            fixed_leg_accrued=fixed_accrued,
            floating_leg_accrued=float_accrued,
            net_accrued=net_accrued,
            dv01=dv01,
            bpv=dv01,
            duration=dur,
            convexity=conv,
            fixed_leg_duration=fixed_leg_dur,
            floating_leg_duration=float_leg_dur,
            weighted_average_life=wal,
        )

    def full_basis_swap_analysis(
        self,
        swap: BasisSwap,
        valuation_date: Optional[datetime] = None,
    ) -> BasisSwapPricingResults:
        """
        Perform full analysis of a basis swap.

        Args:
            swap: Basis swap to analyze
            valuation_date: Valuation date

        Returns:
            BasisSwapPricingResults with all metrics
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        # Update forward rates
        swap.update_forward_rates(self.projection_curve, valuation_date)

        # Calculate leg PVs
        leg1_pv = self.floating_leg_pv(swap.leg1, valuation_date)
        leg2_pv = self.floating_leg_pv(swap.leg2, valuation_date)
        npv = leg2_pv - leg1_pv

        # Par spread
        try:
            par = self.par_spread(swap, valuation_date)
        except ValidationError:
            par = None

        # Risk metrics
        dv01 = self.dv01(swap, valuation_date)
        dur = self.duration(swap, valuation_date)

        return BasisSwapPricingResults(
            npv=npv,
            leg1_pv=leg1_pv,
            leg2_pv=leg2_pv,
            par_spread=par,
            dv01=dv01,
            duration=dur,
        )

    def __repr__(self):
        return f"IRSDiscountEngine(valuation_date={self.pricing_env.valuation_date.date()})"
