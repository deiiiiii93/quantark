"""
Discount-based pricing engine for Floating Rate Notes (FRNs).

Provides pricing, yield calculations (Discount Margin and Simple Margin),
and risk metrics for floating rate bonds.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
import math

from asset.bond.product.couponbond.frn import FloatingRateBond
from asset.bond.schedule.cashflow import FloatingCashFlow
from priceenv import PricingEnvironment
from param.rrf import RateCurve
from util.exceptions import ValidationError, MarketDataError


@dataclass
class FRNPricingResults:
    """
    Results container for FRN pricing.

    Attributes:
        dirty_price: Price including accrued interest
        clean_price: Price excluding accrued interest
        accrued_interest: Accrued interest amount
        discount_margin: Spread over forward that equates price to PV
        simple_margin: Simple yield-based margin
        yield_to_maturity: YTM assuming constant index rate
        effective_duration: Interest rate sensitivity
        spread_duration: Credit spread sensitivity
        weighted_average_life: WAL in years
        current_coupon: Current period coupon rate
        assumed_index_rate: Index rate used for YTM calculation
    """

    dirty_price: float
    clean_price: float
    accrued_interest: float
    discount_margin: Optional[float] = None
    simple_margin: Optional[float] = None
    yield_to_maturity: Optional[float] = None
    effective_duration: Optional[float] = None
    spread_duration: Optional[float] = None
    weighted_average_life: Optional[float] = None
    current_coupon: Optional[float] = None
    assumed_index_rate: Optional[float] = None


class FRNDiscountEngine:
    """
    Pricing engine for Floating Rate Notes.

    Supports:
    - Dirty/Clean price calculation using forward rates
    - Discount Margin (DM) calculation
    - Simple Margin calculation
    - Effective duration and spread duration
    - Weighted average life
    """

    def __init__(self, pricing_env: PricingEnvironment):
        """
        Initialize FRN discount engine.

        Args:
            pricing_env: Pricing environment with rate curve
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")

        if pricing_env.rate_curve is None:
            raise MarketDataError("Rate curve is required for FRN pricing")

        self.pricing_env = pricing_env

    def price(
        self,
        frn: FloatingRateBond,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        spread_adjustment: float = 0.0,
    ) -> float:
        """
        Calculate FRN dirty price (present value including accrued interest).

        Args:
            frn: Floating rate bond to price
            valuation_date: Date to value the bond (default: pricing env date)
            settlement_date: Settlement date for trade (default: valuation_date)
            spread_adjustment: Additional spread for discount margin calculation

        Returns:
            Dirty price (present value of all future cashflows)
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if settlement_date is None:
            settlement_date = valuation_date

        # Check if bond has matured
        if frn.is_expired(valuation_date):
            return 0.0

        # Update forward rates in the FRN
        frn.update_forward_rates(self.pricing_env.rate_curve, valuation_date)

        # Get future floating cashflows
        floating_cfs = frn.get_floating_cashflows(settlement_date)

        if not floating_cfs:
            return 0.0

        # Discount each cashflow
        pv = 0.0
        for cf in floating_cfs:
            # Calculate time to payment
            time_to_payment = (cf.payment_date - valuation_date).days / 365.0

            if time_to_payment < 0:
                continue

            # Get discount rate (base rate + spread adjustment)
            base_discount_rate = self.pricing_env.get_rate(time_to_payment)
            discount_rate = base_discount_rate + spread_adjustment

            # Discount factor
            discount_factor = math.exp(-discount_rate * time_to_payment)

            # Add discounted cashflow
            pv += cf.amount * discount_factor

        # Add discounted principal (at maturity)
        time_to_maturity = (frn.maturity_date - valuation_date).days / 365.0
        if time_to_maturity > 0:
            base_rate = self.pricing_env.get_rate(time_to_maturity)
            discount_rate = base_rate + spread_adjustment
            df_maturity = math.exp(-discount_rate * time_to_maturity)
            pv += frn.denominator * df_maturity

        return pv

    def dirty_price(
        self,
        frn: FloatingRateBond,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate FRN dirty price (alias for price method).

        Args:
            frn: Floating rate bond to price
            valuation_date: Date to value the bond
            settlement_date: Settlement date for trade

        Returns:
            Dirty price
        """
        return self.price(frn, valuation_date, settlement_date)

    def clean_price(
        self,
        frn: FloatingRateBond,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate FRN clean price (dirty price - accrued interest).

        Args:
            frn: Floating rate bond to price
            valuation_date: Date to value the bond
            settlement_date: Settlement date for trade

        Returns:
            Clean price
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if settlement_date is None:
            settlement_date = valuation_date

        dirty = self.dirty_price(frn, valuation_date, settlement_date)
        accrued = frn.calculate_accrued_interest(settlement_date)

        return dirty - accrued

    def accrued_interest(
        self, frn: FloatingRateBond, settlement_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate accrued interest.

        Args:
            frn: Floating rate bond
            settlement_date: Settlement date

        Returns:
            Accrued interest amount
        """
        if settlement_date is None:
            settlement_date = self.pricing_env.valuation_date

        return frn.calculate_accrued_interest(settlement_date)

    def discount_margin(
        self,
        frn: FloatingRateBond,
        market_price: float,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        clean_price: bool = True,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """
        Calculate Discount Margin (DM) using Newton-Raphson iteration.

        The discount margin is the spread over the forward rate curve that
        makes the present value of future cashflows equal to the market price.

        Args:
            frn: Floating rate bond
            market_price: Market price (clean or dirty based on clean_price flag)
            valuation_date: Valuation date
            settlement_date: Settlement date
            clean_price: Whether price is clean price (default: True)
            max_iterations: Maximum iterations for solver
            tolerance: Convergence tolerance

        Returns:
            Discount margin (annualized spread)

        Raises:
            ValidationError: If convergence fails
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if settlement_date is None:
            settlement_date = valuation_date

        if market_price <= 0:
            raise ValidationError(f"Price must be positive, got {market_price}")

        # Convert clean price to dirty price if needed
        if clean_price:
            accrued = frn.calculate_accrued_interest(settlement_date)
            target_price = market_price + accrued
        else:
            target_price = market_price

        # Initial guess: use bond spread
        dm = frn.spread

        # Newton-Raphson iteration
        for iteration in range(max_iterations):
            # Calculate price at current spread
            current_price = self.price(
                frn, valuation_date, settlement_date, spread_adjustment=dm
            )

            # Price difference
            price_diff = current_price - target_price

            if abs(price_diff) < tolerance:
                return dm

            # Calculate numerical derivative (sensitivity to spread)
            bump = 0.0001  # 1bp bump
            bumped_price = self.price(
                frn, valuation_date, settlement_date, spread_adjustment=dm + bump
            )

            dP_dDM = (bumped_price - current_price) / bump

            if abs(dP_dDM) < 1e-10:
                raise ValidationError("Derivative too small for DM calculation")

            # Newton-Raphson update
            dm = dm - price_diff / dP_dDM

            # Sanity check on DM
            if dm < -0.10 or dm > 0.50:
                dm = max(-0.10, min(0.50, dm))

        raise ValidationError(
            f"Discount margin did not converge after {max_iterations} iterations"
        )

    def price_from_yield(
        self,
        frn: FloatingRateBond,
        ytm: float,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        clean_price: bool = True,
        assumed_index_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate FRN price given yield to maturity.

        This is the inverse of yield_to_maturity - given a yield, calculate
        the corresponding FRN price by projecting cashflows with an assumed
        constant index rate and discounting with the given yield.

        Args:
            frn: Floating rate bond
            ytm: Yield to maturity (annualized, continuously compounded)
            valuation_date: Valuation date
            settlement_date: Settlement date
            clean_price: Whether to return clean price (default: True)
            assumed_index_rate: Assumed constant index rate for projection
                               (if None, uses current fixing or forward rate)

        Returns:
            FRN price (clean or dirty based on clean_price flag)
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if settlement_date is None:
            settlement_date = valuation_date

        # Check if bond has matured
        if frn.is_expired(valuation_date):
            return 0.0

        # Update forward rates in the FRN
        frn.update_forward_rates(self.pricing_env.rate_curve, valuation_date)

        # Determine the assumed index rate for projection
        if assumed_index_rate is None:
            current_rate = frn.get_current_coupon_rate(valuation_date)
            if current_rate is not None:
                assumed_index_rate = current_rate - frn.spread
            else:
                assumed_index_rate = self.pricing_env.rate_curve.get_rate(0.25)

        # Get future floating cashflows
        floating_cfs = frn.get_floating_cashflows(settlement_date)

        if not floating_cfs:
            return 0.0

        # Build projected cashflows and discount with given yield
        dirty_price = 0.0
        last_cf_time = 0.0
        last_coupon = 0.0

        for cf in floating_cfs:
            time_to_payment = (cf.payment_date - valuation_date).days / 365.0
            if time_to_payment <= 0:
                continue

            # Calculate coupon amount using assumed rate
            if cf.is_projected:
                total_rate = assumed_index_rate + frn.spread
                if cf.rate_cap is not None:
                    total_rate = min(total_rate, cf.rate_cap)
                if cf.rate_floor is not None:
                    total_rate = max(total_rate, cf.rate_floor)
                coupon_amount = cf.notional * total_rate * cf.day_count_fraction
            else:
                coupon_amount = cf.amount

            # Discount with given yield
            df = math.exp(-ytm * time_to_payment)
            dirty_price += coupon_amount * df

            last_cf_time = time_to_payment
            last_coupon = coupon_amount

        # Add principal at maturity
        time_to_maturity = (frn.maturity_date - valuation_date).days / 365.0
        if time_to_maturity > 0:
            df = math.exp(-ytm * time_to_maturity)
            dirty_price += frn.denominator * df

        if clean_price:
            accrued = frn.calculate_accrued_interest(settlement_date)
            return dirty_price - accrued

        return dirty_price

    def dirty_price_from_yield(
        self,
        frn: FloatingRateBond,
        ytm: float,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        assumed_index_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate dirty price given yield to maturity.

        Args:
            frn: Floating rate bond
            ytm: Yield to maturity (annualized, continuously compounded)
            valuation_date: Valuation date
            settlement_date: Settlement date
            assumed_index_rate: Assumed constant index rate for projection

        Returns:
            Dirty price
        """
        return self.price_from_yield(
            frn,
            ytm,
            valuation_date,
            settlement_date,
            clean_price=False,
            assumed_index_rate=assumed_index_rate,
        )

    def clean_price_from_yield(
        self,
        frn: FloatingRateBond,
        ytm: float,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        assumed_index_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate clean price given yield to maturity.

        Args:
            frn: Floating rate bond
            ytm: Yield to maturity (annualized, continuously compounded)
            valuation_date: Valuation date
            settlement_date: Settlement date
            assumed_index_rate: Assumed constant index rate for projection

        Returns:
            Clean price
        """
        return self.price_from_yield(
            frn,
            ytm,
            valuation_date,
            settlement_date,
            clean_price=True,
            assumed_index_rate=assumed_index_rate,
        )

    def yield_to_maturity(
        self,
        frn: FloatingRateBond,
        market_price: float,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        clean_price: bool = True,
        assumed_index_rate: Optional[float] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
    ) -> float:
        """
        Calculate Yield to Maturity for an FRN.

        For FRNs, YTM is calculated by assuming the floating rate remains
        constant at either:
        1. The provided assumed_index_rate
        2. The current/latest fixing rate
        3. The current forward rate from the curve

        Then finds the single discount rate that equates PV to market price.

        Args:
            frn: Floating rate bond
            market_price: Market price (clean or dirty based on clean_price flag)
            valuation_date: Valuation date
            settlement_date: Settlement date
            clean_price: Whether price is clean price (default: True)
            assumed_index_rate: Assumed constant index rate for projection
                               (if None, uses current fixing or forward rate)
            max_iterations: Maximum iterations for solver
            tolerance: Convergence tolerance

        Returns:
            Yield to maturity (annualized, continuously compounded)

        Raises:
            ValidationError: If convergence fails
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if settlement_date is None:
            settlement_date = valuation_date

        if market_price <= 0:
            raise ValidationError(f"Price must be positive, got {market_price}")

        # Convert clean price to dirty price if needed
        if clean_price:
            accrued = frn.calculate_accrued_interest(settlement_date)
            target_price = market_price + accrued
        else:
            target_price = market_price

        # Update forward rates in the FRN
        frn.update_forward_rates(self.pricing_env.rate_curve, valuation_date)

        # Determine the assumed index rate for projection
        if assumed_index_rate is None:
            # Try to get from latest fixing or current period
            current_rate = frn.get_current_coupon_rate(valuation_date)
            if current_rate is not None:
                # Current coupon includes spread, so extract index rate
                assumed_index_rate = current_rate - frn.spread
            else:
                # Use forward rate at short tenor
                assumed_index_rate = self.pricing_env.rate_curve.get_rate(0.25)

        # Get future floating cashflows and build projected cashflows
        floating_cfs = frn.get_floating_cashflows(settlement_date)

        if not floating_cfs:
            raise ValidationError("No future cashflows to calculate yield")

        # Build list of (time, amount) for all cashflows
        cashflow_data = []
        for cf in floating_cfs:
            time_to_payment = (cf.payment_date - valuation_date).days / 365.0
            if time_to_payment <= 0:
                continue

            # Calculate coupon amount using assumed rate
            if cf.is_projected:
                # Use assumed index rate + spread
                total_rate = assumed_index_rate + frn.spread
                # Apply cap/floor if present
                if cf.rate_cap is not None:
                    total_rate = min(total_rate, cf.rate_cap)
                if cf.rate_floor is not None:
                    total_rate = max(total_rate, cf.rate_floor)
                coupon_amount = cf.notional * total_rate * cf.day_count_fraction
            else:
                # Use actual fixing
                coupon_amount = cf.amount

            cashflow_data.append((time_to_payment, coupon_amount))

        # Add principal at maturity
        time_to_maturity = (frn.maturity_date - valuation_date).days / 365.0
        if time_to_maturity > 0:
            # Find the last cashflow time and add principal there
            if cashflow_data:
                last_time = cashflow_data[-1][0]
                last_coupon = cashflow_data[-1][1]
                cashflow_data[-1] = (last_time, last_coupon + frn.denominator)
            else:
                cashflow_data.append((time_to_maturity, frn.denominator))

        if not cashflow_data:
            raise ValidationError("No valid cashflows for YTM calculation")

        # Initial guess: use current rate plus spread
        ytm = assumed_index_rate + frn.spread

        # Newton-Raphson iteration
        for iteration in range(max_iterations):
            # Calculate price and duration at current yield
            pv = 0.0
            duration = 0.0

            for t, amount in cashflow_data:
                df = math.exp(-ytm * t)
                pv += amount * df
                duration += amount * t * df

            # Check convergence
            price_diff = pv - target_price

            if abs(price_diff) < tolerance:
                return ytm

            # Newton-Raphson update
            # f(y) = PV(y) - target_price
            # f'(y) = -duration
            if abs(duration) < 1e-10:
                raise ValidationError("Duration too small for yield calculation")

            ytm = ytm - price_diff / (-duration)

            # Sanity check on yield
            if ytm < -0.20 or ytm > 1.0:
                ytm = max(-0.20, min(1.0, ytm))

        raise ValidationError(
            f"Yield to maturity did not converge after {max_iterations} iterations"
        )

    def simple_margin(
        self,
        frn: FloatingRateBond,
        market_price: float,
        valuation_date: Optional[datetime] = None,
        clean_price: bool = True,
    ) -> float:
        """
        Calculate Simple Margin.

        Simple Margin is an approximation:
        SM = (100 - Price) / WAL + Quoted Spread

        Where WAL is weighted average life.

        Args:
            frn: Floating rate bond
            market_price: Market price (as % of par, e.g., 99.5)
            valuation_date: Valuation date
            clean_price: Whether price is clean price

        Returns:
            Simple margin (annualized)
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        # Calculate weighted average life
        wal = self.weighted_average_life(frn, valuation_date)

        if wal <= 0:
            return frn.spread

        # Convert price to percentage of par
        if clean_price:
            price_pct = (market_price / frn.denominator) * 100
        else:
            accrued = frn.calculate_accrued_interest(valuation_date)
            clean = market_price - accrued
            price_pct = (clean / frn.denominator) * 100

        # Simple margin formula
        # (100 - Price) / WAL gives annualized capital gain/loss
        capital_component = (100.0 - price_pct) / (100.0 * wal)

        return capital_component + frn.spread

    def weighted_average_life(
        self, frn: FloatingRateBond, valuation_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate Weighted Average Life (WAL).

        WAL = sum(t_i * P_i) / sum(P_i)

        Where t_i is time to principal payment and P_i is principal amount.
        For bullet FRNs, this equals time to maturity.

        Args:
            frn: Floating rate bond
            valuation_date: Valuation date

        Returns:
            Weighted average life in years
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        # For bullet FRNs (single principal payment at maturity)
        time_to_maturity = (frn.maturity_date - valuation_date).days / 365.0

        return max(0.0, time_to_maturity)

    def effective_duration(
        self,
        frn: FloatingRateBond,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        rate_bump: float = 0.0001,
    ) -> float:
        """
        Calculate effective duration (sensitivity to parallel rate shift).

        For FRNs, effective duration is typically very low since cashflows
        reset to market rates. The duration is mainly from the time to
        next reset.

        Args:
            frn: Floating rate bond
            valuation_date: Valuation date
            settlement_date: Settlement date
            rate_bump: Size of rate bump (default: 1bp)

        Returns:
            Effective duration
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if settlement_date is None:
            settlement_date = valuation_date

        base_price = self.price(frn, valuation_date, settlement_date)

        if base_price <= 0:
            return 0.0

        # Price with rates bumped up
        from param.rrf import FlatRateCurve

        original_curve = self.pricing_env.rate_curve

        # Create bumped curve (simple flat bump)
        base_rate = original_curve.get_rate(1.0)

        up_curve = FlatRateCurve(rate=base_rate + rate_bump)
        down_curve = FlatRateCurve(rate=base_rate - rate_bump)

        # Price with up curve
        self.pricing_env.rate_curve = up_curve
        price_up = self.price(frn, valuation_date, settlement_date)

        # Price with down curve
        self.pricing_env.rate_curve = down_curve
        price_down = self.price(frn, valuation_date, settlement_date)

        # Restore original curve
        self.pricing_env.rate_curve = original_curve

        # Effective duration = -(P+ - P-) / (2 * dY * P0)
        duration = -(price_up - price_down) / (2 * rate_bump * base_price)

        return duration

    def spread_duration(
        self,
        frn: FloatingRateBond,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        spread_bump: float = 0.0001,
    ) -> float:
        """
        Calculate spread duration (sensitivity to credit spread changes).

        Spread duration measures the price sensitivity to changes in the
        discount margin / credit spread. For FRNs, this is approximately
        equal to the weighted average life.

        Args:
            frn: Floating rate bond
            valuation_date: Valuation date
            settlement_date: Settlement date
            spread_bump: Size of spread bump (default: 1bp)

        Returns:
            Spread duration
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if settlement_date is None:
            settlement_date = valuation_date

        base_price = self.price(frn, valuation_date, settlement_date)

        if base_price <= 0:
            return 0.0

        # Price with spread bumped up and down
        price_up = self.price(
            frn, valuation_date, settlement_date, spread_adjustment=spread_bump
        )
        price_down = self.price(
            frn, valuation_date, settlement_date, spread_adjustment=-spread_bump
        )

        # Spread duration = -(P+ - P-) / (2 * dS * P0)
        spread_dur = -(price_up - price_down) / (2 * spread_bump * base_price)

        return spread_dur

    def dv01(
        self,
        frn: FloatingRateBond,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate DV01 (dollar value of one basis point).

        Args:
            frn: Floating rate bond
            valuation_date: Valuation date
            settlement_date: Settlement date

        Returns:
            DV01 (price change per basis point)
        """
        eff_dur = self.effective_duration(frn, valuation_date, settlement_date)
        price = self.price(frn, valuation_date, settlement_date)

        return eff_dur * price * 0.0001

    def cs01(
        self,
        frn: FloatingRateBond,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate CS01 (credit spread 01 - price change per 1bp spread change).

        Args:
            frn: Floating rate bond
            valuation_date: Valuation date
            settlement_date: Settlement date

        Returns:
            CS01
        """
        spread_dur = self.spread_duration(frn, valuation_date, settlement_date)
        price = self.price(frn, valuation_date, settlement_date)

        return spread_dur * price * 0.0001

    def full_analysis(
        self,
        frn: FloatingRateBond,
        market_price: Optional[float] = None,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        clean_price: bool = True,
        assumed_index_rate: Optional[float] = None,
    ) -> FRNPricingResults:
        """
        Perform full analysis of an FRN.

        Args:
            frn: Floating rate bond
            market_price: Market price (if provided, calculates margins and YTM)
            valuation_date: Valuation date
            settlement_date: Settlement date
            clean_price: Whether market_price is clean
            assumed_index_rate: Index rate to assume for YTM calculation

        Returns:
            FRNPricingResults with all metrics
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if settlement_date is None:
            settlement_date = valuation_date

        dirty = self.dirty_price(frn, valuation_date, settlement_date)
        accrued = frn.calculate_accrued_interest(settlement_date)
        clean = dirty - accrued

        # Calculate margins and YTM if market price provided
        dm = None
        sm = None
        ytm = None
        index_rate_used = assumed_index_rate

        if market_price is not None:
            try:
                dm = self.discount_margin(
                    frn, market_price, valuation_date, settlement_date, clean_price
                )
            except ValidationError:
                pass

            sm = self.simple_margin(frn, market_price, valuation_date, clean_price)

            try:
                ytm = self.yield_to_maturity(
                    frn,
                    market_price,
                    valuation_date,
                    settlement_date,
                    clean_price,
                    assumed_index_rate,
                )
                # Determine the index rate that was actually used
                if index_rate_used is None:
                    current_coupon = frn.get_current_coupon_rate(valuation_date)
                    if current_coupon is not None:
                        index_rate_used = current_coupon - frn.spread
                    else:
                        index_rate_used = self.pricing_env.rate_curve.get_rate(0.25)
            except ValidationError:
                pass

        # Risk metrics
        eff_dur = self.effective_duration(frn, valuation_date, settlement_date)
        spread_dur = self.spread_duration(frn, valuation_date, settlement_date)
        wal = self.weighted_average_life(frn, valuation_date)
        current_coupon = frn.get_current_coupon_rate(valuation_date)

        return FRNPricingResults(
            dirty_price=dirty,
            clean_price=clean,
            accrued_interest=accrued,
            discount_margin=dm,
            simple_margin=sm,
            yield_to_maturity=ytm,
            effective_duration=eff_dur,
            spread_duration=spread_dur,
            weighted_average_life=wal,
            current_coupon=current_coupon,
            assumed_index_rate=index_rate_used,
        )

    def __repr__(self):
        return f"FRNDiscountEngine(valuation_date={self.pricing_env.valuation_date.date()})"
