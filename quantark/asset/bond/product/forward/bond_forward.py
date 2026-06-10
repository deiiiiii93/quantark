"""
Bond forward product implementation.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from asset.bond.product.forward.base_bond_forward import BaseBondForward
from asset.bond.product.couponbond.fixed_bond import FixedBond
from util.exceptions import ValidationError


@dataclass
class BondForward(BaseBondForward):
    """
    Forward contract on a fixed-rate bond.

    A bond forward is an OTC agreement to buy/sell a specific bond at a future
    date at a predetermined price. The forward price incorporates carry costs
    (financing via repo) and any coupon income during the forward period.

    Pricing Model:
        Forward Dirty Price = Spot Dirty Price * exp(repo_rate * T) - FV(coupons)
        Invoice Price = Forward Dirty Price + Accrued at Delivery

    Where:
        - Spot Dirty Price: Current bond price including accrued interest
        - repo_rate: Financing rate (repo rate)
        - T: Time to delivery
        - FV(coupons): Future value of any coupons received before delivery

    Attributes:
        underlying: The FixedBond being delivered
        delivery_date: Date when bond is delivered
        forward_price: Agreed forward clean price (None for theoretical pricing)
        repo_rate: Repo/financing rate for carry calculation
        contract_size: Denominator per contract (default: 100,000)
        is_long: True if long the forward (buying the bond), False if short

    Example:
        >>> from datetime import datetime
        >>> from asset.bond.product.couponbond.fixed_bond import FixedBond
        >>> from util.enum import PaymentFrequency
        >>> from util.calendar import DayCountConvention
        >>>
        >>> bond = FixedBond(
        ...     issue_date=datetime(2024, 1, 15),
        ...     maturity_date=datetime(2034, 1, 15),
        ...     denominator=100.0,
        ...     coupon_rate=0.05,
        ...     payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        ...     day_count_convention=DayCountConvention.ACT_ACT_ISDA
        ... )
        >>>
        >>> forward = BondForward(
        ...     underlying=bond,
        ...     delivery_date=datetime(2024, 6, 15),
        ...     repo_rate=0.045,
        ...     is_long=True
        ... )
    """

    underlying: FixedBond = field(default=None)
    forward_price: Optional[float] = None
    repo_rate: float = 0.0
    is_long: bool = True

    def __post_init__(self):
        """Initialize and validate the forward contract."""
        if self.underlying is None:
            raise ValidationError("Underlying bond is required")
        self.validate()

    def validate(self) -> None:
        """
        Validate forward contract parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        if self.underlying is None:
            raise ValidationError("Underlying bond is required")

        if self.delivery_date >= self.underlying.maturity_date:
            raise ValidationError(
                f"Delivery date {self.delivery_date} must be before "
                f"bond maturity {self.underlying.maturity_date}"
            )

        if self.delivery_date <= self.underlying.issue_date:
            raise ValidationError(
                f"Delivery date {self.delivery_date} must be after "
                f"bond issue date {self.underlying.issue_date}"
            )

        if self.contract_size <= 0:
            raise ValidationError(
                f"Contract size must be positive, got {self.contract_size}"
            )

        if self.forward_price is not None and self.forward_price <= 0:
            raise ValidationError(
                f"Forward price must be positive if specified, got {self.forward_price}"
            )

        if not math.isfinite(self.repo_rate):
            raise ValidationError(f"Repo rate must be finite, got {self.repo_rate}")

    def get_delivery_date(self) -> datetime:
        """Get the delivery date of the forward."""
        return self.delivery_date

    def get_underlying(self) -> FixedBond:
        """Get the underlying bond."""
        return self.underlying

    def get_accrued_at_delivery(self) -> float:
        """
        Calculate accrued interest on the delivery date.

        Returns:
            Accrued interest at delivery
        """
        return self.underlying.calculate_accrued_interest(self.delivery_date)

    def get_coupons_before_delivery(self, valuation_date: datetime) -> list:
        """
        Get all coupon payments between valuation date and delivery date.

        Args:
            valuation_date: Current valuation date

        Returns:
            List of cashflows between valuation and delivery
        """
        all_cashflows = self.underlying.get_all_cashflows()

        coupons = []
        for cf in all_cashflows:
            # Include coupons that pay after valuation but before or on delivery
            if valuation_date < cf.payment_date <= self.delivery_date:
                # Exclude principal repayment (at maturity)
                coupon_amount = cf.amount
                if cf.payment_date == self.underlying.maturity_date:
                    coupon_amount = cf.amount - self.underlying.get_denominator()
                if coupon_amount > 0:
                    coupons.append(cf)

        return coupons

    def calculate_forward_dirty_price(
        self, spot_dirty_price: float, valuation_date: datetime
    ) -> float:
        """
        Calculate theoretical forward dirty price.

        Forward Dirty Price = Spot Dirty * exp(r*T) - FV(coupons)

        The forward price accounts for:
        1. Financing cost of carrying the bond (repo rate)
        2. Coupon income received before delivery (reinvested at repo rate)

        Args:
            spot_dirty_price: Current bond dirty price
            valuation_date: Current valuation date

        Returns:
            Theoretical forward dirty price
        """
        time_to_delivery = self.get_time_to_delivery(valuation_date)

        if time_to_delivery <= 0:
            return spot_dirty_price

        # Carry cost
        carry_factor = math.exp(self.repo_rate * time_to_delivery)
        carried_price = spot_dirty_price * carry_factor

        # Subtract future value of coupons received before delivery
        coupons = self.get_coupons_before_delivery(valuation_date)
        fv_coupons = 0.0

        for cf in coupons:
            # Time from coupon payment to delivery
            time_to_delivery_from_coupon = (
                self.delivery_date - cf.payment_date
            ).days / 365.0

            # Future value of coupon at delivery (reinvested at repo)
            fv_coupon = cf.amount * math.exp(
                self.repo_rate * time_to_delivery_from_coupon
            )
            fv_coupons += fv_coupon

        forward_dirty = carried_price - fv_coupons

        return forward_dirty

    def calculate_forward_clean_price(
        self, spot_dirty_price: float, valuation_date: datetime
    ) -> float:
        """
        Calculate theoretical forward clean price.

        Forward Clean = Forward Dirty - Accrued at Delivery

        Args:
            spot_dirty_price: Current bond dirty price
            valuation_date: Current valuation date

        Returns:
            Theoretical forward clean price
        """
        forward_dirty = self.calculate_forward_dirty_price(
            spot_dirty_price, valuation_date
        )
        accrued_at_delivery = self.get_accrued_at_delivery()

        return forward_dirty - accrued_at_delivery

    def calculate_invoice_price(
        self, spot_dirty_price: float, valuation_date: datetime
    ) -> float:
        """
        Calculate the invoice price (total amount paid at delivery).

        Invoice Price = Forward Dirty Price

        Note: For bond forwards, the invoice price typically equals the
        forward dirty price. The convention varies by market.

        Args:
            spot_dirty_price: Current bond dirty price
            valuation_date: Current valuation date

        Returns:
            Invoice price
        """
        return self.calculate_forward_dirty_price(spot_dirty_price, valuation_date)

    def calculate_forward_value(
        self, spot_dirty_price: float, valuation_date: datetime, discount_factor: float
    ) -> float:
        """
        Calculate the mark-to-market value of the forward contract.

        Value = DF * (Theoretical Forward - Contract Forward) * direction

        Where direction is +1 for long, -1 for short.

        Args:
            spot_dirty_price: Current bond dirty price
            valuation_date: Current valuation date
            discount_factor: Discount factor from valuation to delivery

        Returns:
            Forward contract value (positive = profit, negative = loss)
        """
        if self.forward_price is None:
            # No contracted price, return zero value
            return 0.0

        theoretical_forward_clean = self.calculate_forward_clean_price(
            spot_dirty_price, valuation_date
        )

        # Contract value based on clean price difference
        price_diff = theoretical_forward_clean - self.forward_price
        direction = 1.0 if self.is_long else -1.0

        # Scale by contract size and discount to present value
        value = discount_factor * price_diff * (self.contract_size / 100.0) * direction

        return value

    def get_implied_repo_rate(
        self,
        spot_dirty_price: float,
        forward_clean_price: float,
        valuation_date: datetime,
    ) -> float:
        """
        Calculate implied repo rate from spot and forward prices.

        Solves for r in: Forward Dirty = Spot Dirty * exp(r*T) - FV(coupons)

        Args:
            spot_dirty_price: Current bond dirty price
            forward_clean_price: Observed forward clean price
            valuation_date: Current valuation date

        Returns:
            Implied repo rate
        """
        time_to_delivery = self.get_time_to_delivery(valuation_date)

        if time_to_delivery <= 0:
            return 0.0

        # Forward dirty from forward clean
        forward_dirty = forward_clean_price + self.get_accrued_at_delivery()

        # Get coupons - need to iterate to solve for repo rate
        # For simplicity, assume coupons are small relative to price
        # and solve approximately
        coupons = self.get_coupons_before_delivery(valuation_date)
        total_coupon = sum(cf.amount for cf in coupons)

        # Approximate: forward_dirty + total_coupon ≈ spot_dirty * exp(r*T)
        adjusted_forward = forward_dirty + total_coupon

        if spot_dirty_price <= 0:
            return 0.0

        implied_repo = math.log(adjusted_forward / spot_dirty_price) / time_to_delivery

        return implied_repo

    def get_denominator(self) -> float:
        """Get the minimum tradable notional (denominator) per contract."""
        return self.contract_size

    def __repr__(self):
        direction = "Long" if self.is_long else "Short"
        price_str = (
            f", fwd_price={self.forward_price:.4f}" if self.forward_price else ""
        )
        return (
            f"BondForward({direction}, "
            f"delivery={self.delivery_date.date()}, "
            f"repo={self.repo_rate:.4f}{price_str}, "
            f"underlying={self.underlying})"
        )
