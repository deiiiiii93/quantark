"""
Pricing engine for bond forward contracts.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from quantark.asset.bond.product.forward.bond_forward import BondForward
from quantark.asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError, PricingError


@dataclass
class BondForwardResults:
    """
    Results from bond forward pricing.

    Attributes:
        forward_value: Mark-to-market value of the forward
        forward_clean_price: Theoretical forward clean price
        forward_dirty_price: Theoretical forward dirty price
        invoice_price: Total invoice amount at delivery
        spot_dirty_price: Current bond dirty price
        spot_clean_price: Current bond clean price
        accrued_at_spot: Accrued interest at valuation
        accrued_at_delivery: Accrued interest at delivery
        time_to_delivery: Time to delivery in years
        implied_repo_rate: Implied repo rate (if forward price set)
    """

    forward_value: float
    forward_clean_price: float
    forward_dirty_price: float
    invoice_price: float
    spot_dirty_price: float
    spot_clean_price: float
    accrued_at_spot: float
    accrued_at_delivery: float
    time_to_delivery: float
    implied_repo_rate: Optional[float] = None


class BondForwardEngine:
    """
    Pricing engine for bond forward contracts.

    This engine prices bond forwards by:
    1. Pricing the underlying bond using BondDiscountEngine
    2. Calculating the forward price using cost-of-carry (repo rate)
    3. Computing the mark-to-market value vs contracted price

    Greeks are calculated using finite difference bumping:
    - DV01: Dollar value of 1 basis point rate move
    - Modified Duration: Price sensitivity to yield
    - Convexity: Second-order price sensitivity
    - Repo Sensitivity: Sensitivity to repo rate changes

    Attributes:
        pricing_env: Pricing environment with market data
        bond_engine: Underlying bond pricing engine
        bump_size: Basis point bump for Greeks (default: 1bp)
    """

    def __init__(self, pricing_env: PricingEnvironment, bump_size: float = 0.0001):
        """
        Initialize bond forward engine.

        Args:
            pricing_env: Pricing environment with rate curve
            bump_size: Bump size for finite difference Greeks (default: 1bp)
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")

        self.pricing_env = pricing_env
        self.bond_engine = BondDiscountEngine(pricing_env)
        self.bump_size = bump_size

    def price(
        self, forward: BondForward, valuation_date: Optional[datetime] = None
    ) -> BondForwardResults:
        """
        Price a bond forward contract.

        Args:
            forward: Bond forward contract to price
            valuation_date: Valuation date (default: pricing env date)

        Returns:
            BondForwardResults with pricing details

        Raises:
            PricingError: If pricing fails
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        # Check if expired
        if forward.is_expired(valuation_date):
            raise PricingError("Forward contract has expired")

        # Price underlying bond
        underlying = forward.underlying
        spot_dirty_price = self.bond_engine.dirty_price(
            underlying, valuation_date, valuation_date
        )
        spot_clean_price = self.bond_engine.clean_price(
            underlying, valuation_date, valuation_date
        )
        accrued_at_spot = underlying.calculate_accrued_interest(valuation_date)

        # Calculate forward prices
        forward_dirty_price = forward.calculate_forward_dirty_price(
            spot_dirty_price, valuation_date
        )
        forward_clean_price = forward.calculate_forward_clean_price(
            spot_dirty_price, valuation_date
        )
        invoice_price = forward.calculate_invoice_price(
            spot_dirty_price, valuation_date
        )
        accrued_at_delivery = forward.get_accrued_at_delivery()

        # Time to delivery
        time_to_delivery = forward.get_time_to_delivery(valuation_date)

        # Discount factor from valuation to delivery
        discount_factor = self.pricing_env.get_discount_factor(time_to_delivery)

        # Calculate forward value
        forward_value = forward.calculate_forward_value(
            spot_dirty_price, valuation_date, discount_factor
        )

        # Implied repo rate if forward price is set
        implied_repo = None
        if forward.forward_price is not None:
            implied_repo = forward.get_implied_repo_rate(
                spot_dirty_price, forward.forward_price, valuation_date
            )

        return BondForwardResults(
            forward_value=forward_value,
            forward_clean_price=forward_clean_price,
            forward_dirty_price=forward_dirty_price,
            invoice_price=invoice_price,
            spot_dirty_price=spot_dirty_price,
            spot_clean_price=spot_clean_price,
            accrued_at_spot=accrued_at_spot,
            accrued_at_delivery=accrued_at_delivery,
            time_to_delivery=time_to_delivery,
            implied_repo_rate=implied_repo,
        )

    def calculate_greeks(
        self, forward: BondForward, valuation_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate Greeks for bond forward.

        Greeks calculated:
        - forward_value: Current mark-to-market value
        - forward_clean_price: Theoretical forward clean price
        - dv01: Dollar value of 1bp parallel shift in rates
        - modified_duration: Effective modified duration
        - convexity: Second-order rate sensitivity
        - repo_sensitivity: Sensitivity to repo rate (per 1bp)
        - carry: Daily carry (theta equivalent)

        Args:
            forward: Bond forward contract
            valuation_date: Valuation date

        Returns:
            Dictionary of Greeks
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        # Base pricing
        base_results = self.price(forward, valuation_date)

        greeks = {
            "forward_value": base_results.forward_value,
            "forward_clean_price": base_results.forward_clean_price,
            "forward_dirty_price": base_results.forward_dirty_price,
            "spot_dirty_price": base_results.spot_dirty_price,
        }

        # DV01: Bump rates up by 1bp
        greeks["dv01"] = self._calculate_dv01(forward, valuation_date, base_results)

        # Modified Duration
        greeks["modified_duration"] = self._calculate_modified_duration(
            forward, valuation_date, base_results
        )

        # Convexity
        greeks["convexity"] = self._calculate_convexity(
            forward, valuation_date, base_results
        )

        # Repo sensitivity
        greeks["repo_sensitivity"] = self._calculate_repo_sensitivity(
            forward, valuation_date, base_results
        )

        # Carry (theta equivalent) - value change per day
        greeks["carry"] = self._calculate_carry(forward, valuation_date, base_results)

        return greeks

    def _calculate_dv01(
        self,
        forward: BondForward,
        valuation_date: datetime,
        base_results: BondForwardResults,
    ) -> float:
        """Calculate DV01 using parallel rate bump."""
        from quantark.param.rrf.rate_curve import FlatRateCurve

        # Get base rate and create bumped curve
        base_rate = self.pricing_env.rate_curve.get_rate(1.0)
        bumped_rate = base_rate + self.bump_size
        bumped_curve = FlatRateCurve(rate=bumped_rate)

        env_up = PricingEnvironment(
            rate_curve=bumped_curve,
            valuation_date=self.pricing_env.valuation_date,
            spot_quote=self.pricing_env.spot_quote,
            vol_surface=self.pricing_env.vol_surface,
            div_yield=self.pricing_env.div_yield,
        )

        engine_up = BondForwardEngine(env_up, self.bump_size)
        results_up = engine_up.price(forward, valuation_date)

        # DV01 = change in forward dirty price for 1bp rate increase
        dv01 = base_results.forward_dirty_price - results_up.forward_dirty_price

        return dv01

    def _calculate_modified_duration(
        self,
        forward: BondForward,
        valuation_date: datetime,
        base_results: BondForwardResults,
    ) -> float:
        """Calculate modified duration of forward."""
        if base_results.forward_dirty_price == 0:
            return 0.0

        dv01 = self._calculate_dv01(forward, valuation_date, base_results)

        # Modified Duration = (DV01 / Price) / bump_size
        # = DV01 / (Price * 0.0001) = DV01 * 10000 / Price
        mod_dur = dv01 / (base_results.forward_dirty_price * self.bump_size)

        return mod_dur

    def _calculate_convexity(
        self,
        forward: BondForward,
        valuation_date: datetime,
        base_results: BondForwardResults,
    ) -> float:
        """Calculate convexity using central difference."""
        from quantark.param.rrf.rate_curve import FlatRateCurve

        if base_results.forward_dirty_price == 0:
            return 0.0

        # Get base rate
        base_rate = self.pricing_env.rate_curve.get_rate(1.0)

        # Bump up
        bumped_rate_up = base_rate + self.bump_size
        curve_up = FlatRateCurve(rate=bumped_rate_up)
        env_up = PricingEnvironment(
            rate_curve=curve_up,
            valuation_date=self.pricing_env.valuation_date,
            spot_quote=self.pricing_env.spot_quote,
            vol_surface=self.pricing_env.vol_surface,
            div_yield=self.pricing_env.div_yield,
        )
        engine_up = BondForwardEngine(env_up, self.bump_size)
        results_up = engine_up.price(forward, valuation_date)

        # Bump down
        bumped_rate_down = base_rate - self.bump_size
        curve_down = FlatRateCurve(rate=bumped_rate_down)
        env_down = PricingEnvironment(
            rate_curve=curve_down,
            valuation_date=self.pricing_env.valuation_date,
            spot_quote=self.pricing_env.spot_quote,
            vol_surface=self.pricing_env.vol_surface,
            div_yield=self.pricing_env.div_yield,
        )
        engine_down = BondForwardEngine(env_down, self.bump_size)
        results_down = engine_down.price(forward, valuation_date)

        # Convexity = (P_up - 2*P_base + P_down) / (P_base * bump^2)
        price_up = results_up.forward_dirty_price
        price_down = results_down.forward_dirty_price
        price_base = base_results.forward_dirty_price

        convexity = (price_up - 2 * price_base + price_down) / (
            price_base * self.bump_size * self.bump_size
        )

        return convexity

    def _calculate_repo_sensitivity(
        self,
        forward: BondForward,
        valuation_date: datetime,
        base_results: BondForwardResults,
    ) -> float:
        """Calculate sensitivity to repo rate (per 1bp)."""
        from copy import deepcopy

        # Create forward with bumped repo rate
        forward_bumped = deepcopy(forward)
        forward_bumped.repo_rate = forward.repo_rate + self.bump_size

        results_up = self.price(forward_bumped, valuation_date)

        # Sensitivity = change in forward price for 1bp repo increase
        repo_sens = results_up.forward_dirty_price - base_results.forward_dirty_price

        return repo_sens

    def _calculate_carry(
        self,
        forward: BondForward,
        valuation_date: datetime,
        base_results: BondForwardResults,
    ) -> float:
        """Calculate daily carry (value change per day)."""
        from datetime import timedelta

        # Price forward one day forward
        next_day = valuation_date + timedelta(days=1)

        if forward.is_expired(next_day):
            return 0.0

        try:
            results_next = self.price(forward, next_day)
            carry = results_next.forward_value - base_results.forward_value
            return carry
        except PricingError:
            return 0.0

    def calculate_basis_point_value(
        self, forward: BondForward, valuation_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate basis point value (BPV) of the forward.

        BPV = DV01 * Contract Size / 100

        Args:
            forward: Bond forward contract
            valuation_date: Valuation date

        Returns:
            Basis point value in dollars
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        base_results = self.price(forward, valuation_date)
        dv01 = self._calculate_dv01(forward, valuation_date, base_results)

        # Scale by contract size relative to underlying bond denominator.
        denominator = forward.underlying.get_denominator()
        if denominator <= 0:
            raise ValidationError(
                f"Denominator must be positive, got {denominator}"
            )
        bpv = dv01 * (forward.contract_size / denominator)

        return bpv

    def __repr__(self):
        return f"BondForwardEngine(valuation_date={self.pricing_env.valuation_date.date()})"
