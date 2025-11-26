"""
Bond futures product implementation with CTD (Cheapest to Deliver) logic.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from asset.bond.product.forward.base_bond_forward import BaseBondForward
from asset.bond.product.couponbond.fixed_bond import FixedBond
from util.exceptions import ValidationError


@dataclass
class DeliverableBond:
    """
    A bond in the deliverable basket with its conversion factor.

    Attributes:
        bond: The FixedBond that can be delivered
        conversion_factor: Exchange conversion factor (None = auto-calculate)
    """

    bond: FixedBond
    conversion_factor: Optional[float] = None


@dataclass
class BondFutures(BaseBondForward):
    """
    Bond futures contract with Cheapest-to-Deliver (CTD) logic.

    Bond futures allow delivery of any bond from a basket of eligible bonds.
    Each bond has a conversion factor (CF) that adjusts for coupon and maturity
    differences. The CTD bond is the one with the lowest delivery cost.

    Key Concepts:
        - Conversion Factor (CF): Adjusts bond price to 6% notional equivalent
        - Gross Basis: Bond Price - Futures Price × CF
        - Net Basis: Gross Basis - Carry
        - CTD Bond: Bond with lowest net basis (or highest implied repo)

    Invoice Price at Delivery:
        Invoice = Futures Price × CF + Accrued Interest

    Attributes:
        delivery_date: Futures contract delivery/expiration date
        deliverable_basket: List of DeliverableBond objects
        contract_size: Notional per contract (default: 100,000)
        futures_price: Current futures price (for CTD analysis)
        tick_size: Minimum price movement (default: 1/32 of 1%)
        notional_coupon: Coupon rate for CF calculation (default: 6%)

    Example:
        >>> from datetime import datetime
        >>> from asset.bond.product.couponbond.fixed_bond import FixedBond
        >>> from util.enum import PaymentFrequency
        >>> from util.calendar import DayCountConvention
        >>>
        >>> # Create deliverable bonds
        >>> bond1 = FixedBond(
        ...     issue_date=datetime(2020, 1, 15),
        ...     maturity_date=datetime(2030, 1, 15),
        ...     notional=100.0,
        ...     coupon_rate=0.04,
        ...     payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        ...     day_count_convention=DayCountConvention.ACT_ACT_ISDA
        ... )
        >>> bond2 = FixedBond(
        ...     issue_date=datetime(2021, 6, 15),
        ...     maturity_date=datetime(2031, 6, 15),
        ...     notional=100.0,
        ...     coupon_rate=0.05,
        ...     payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        ...     day_count_convention=DayCountConvention.ACT_ACT_ISDA
        ... )
        >>>
        >>> futures = BondFutures(
        ...     delivery_date=datetime(2025, 3, 15),
        ...     deliverable_basket=[
        ...         DeliverableBond(bond1),
        ...         DeliverableBond(bond2, conversion_factor=0.95)
        ...     ],
        ...     futures_price=115.50
        ... )
    """

    deliverable_basket: List[DeliverableBond] = field(default_factory=list)
    futures_price: Optional[float] = None
    tick_size: float = 1.0 / 32.0
    notional_coupon: float = 0.06  # 6% for CBOT/CME

    # Cached conversion factors
    _conversion_factors: Dict[int, float] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Initialize and validate the futures contract."""
        if self._conversion_factors is None:
            self._conversion_factors = {}
        self.validate()
        self._compute_conversion_factors()

    def validate(self) -> None:
        """
        Validate futures contract parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        if not self.deliverable_basket:
            raise ValidationError("Deliverable basket cannot be empty")

        if self.contract_size <= 0:
            raise ValidationError(
                f"Contract size must be positive, got {self.contract_size}"
            )

        if self.futures_price is not None and self.futures_price <= 0:
            raise ValidationError(
                f"Futures price must be positive if specified, got {self.futures_price}"
            )

        if self.tick_size <= 0:
            raise ValidationError(f"Tick size must be positive, got {self.tick_size}")

        if self.notional_coupon < 0:
            raise ValidationError(
                f"Notional coupon must be non-negative, got {self.notional_coupon}"
            )

        # Validate each deliverable bond
        for i, db in enumerate(self.deliverable_basket):
            if db.bond is None:
                raise ValidationError(f"Bond at index {i} is None")

            if self.delivery_date >= db.bond.maturity_date:
                raise ValidationError(
                    f"Delivery date {self.delivery_date} must be before "
                    f"bond maturity {db.bond.maturity_date} for bond {i}"
                )

            if db.conversion_factor is not None and db.conversion_factor <= 0:
                raise ValidationError(
                    f"Conversion factor must be positive, got {db.conversion_factor} "
                    f"for bond {i}"
                )

    def get_delivery_date(self) -> datetime:
        """Get the delivery date of the futures."""
        return self.delivery_date

    def _compute_conversion_factors(self) -> None:
        """
        Compute conversion factors for all deliverable bonds.

        Uses provided CFs if available, otherwise calculates using
        standard exchange formula.
        """
        self._conversion_factors = {}

        for i, db in enumerate(self.deliverable_basket):
            if db.conversion_factor is not None:
                # Use provided conversion factor
                self._conversion_factors[i] = db.conversion_factor
            else:
                # Calculate conversion factor
                cf = self.calculate_conversion_factor(db.bond)
                self._conversion_factors[i] = cf
                db.conversion_factor = cf

    def calculate_conversion_factor(self, bond: FixedBond) -> float:
        """
        Calculate conversion factor using CBOT/CME methodology.

        The conversion factor is the price of the bond (per $1 face value)
        to yield the notional coupon rate (typically 6%), assuming the bond
        is delivered on the first day of the delivery month.

        Formula (simplified for semi-annual bonds):
            CF = (1/1.03^n) * [c/2 * (1 - 1/1.03^(2n)) / 0.03 + 1]

        Where:
            n = years to maturity from delivery (rounded to nearest quarter)
            c = annual coupon rate

        Args:
            bond: The bond to calculate CF for

        Returns:
            Conversion factor
        """
        # Time from delivery to maturity
        years_to_maturity = (bond.maturity_date - self.delivery_date).days / 365.0

        # Round to nearest quarter (CBOT convention)
        quarters = round(years_to_maturity * 4) / 4.0
        n = max(quarters, 0.25)  # Minimum quarter year

        # Semi-annual yield equivalent of notional coupon
        y = self.notional_coupon / 2.0  # 3% per semi-annual period

        # Number of semi-annual periods
        periods = int(n * 2)

        if periods <= 0:
            return 1.0

        # Discount factor for n years at notional coupon
        df = 1.0 / ((1.0 + y) ** periods)

        # Coupon rate
        c = bond.coupon_rate

        # Present value of coupons
        if y > 0:
            coupon_pv = (c / 2.0) * (1.0 - df) / y
        else:
            coupon_pv = (c / 2.0) * periods

        # Present value of principal
        principal_pv = df

        # Total conversion factor
        cf = coupon_pv + principal_pv

        # Round to 4 decimal places (exchange convention)
        return round(cf, 4)

    def get_conversion_factor(self, bond_index: int) -> float:
        """
        Get the conversion factor for a specific bond.

        Args:
            bond_index: Index of bond in deliverable basket

        Returns:
            Conversion factor

        Raises:
            ValidationError: If index is out of range
        """
        if bond_index < 0 or bond_index >= len(self.deliverable_basket):
            raise ValidationError(
                f"Bond index {bond_index} out of range [0, {len(self.deliverable_basket)})"
            )

        return self._conversion_factors.get(bond_index, 1.0)

    def calculate_gross_basis(self, bond_index: int, bond_dirty_price: float) -> float:
        """
        Calculate gross basis for a deliverable bond.

        Gross Basis = Bond Dirty Price - Futures Price × Conversion Factor

        The gross basis represents the cost of buying the bond and delivering
        it into the futures contract.

        Args:
            bond_index: Index of bond in deliverable basket
            bond_dirty_price: Current dirty price of the bond

        Returns:
            Gross basis

        Raises:
            ValidationError: If futures price not set
        """
        if self.futures_price is None:
            raise ValidationError("Futures price must be set for basis calculation")

        cf = self.get_conversion_factor(bond_index)

        return bond_dirty_price - self.futures_price * cf

    def calculate_net_basis(
        self, bond_index: int, bond_dirty_price: float, carry: float
    ) -> float:
        """
        Calculate net basis for a deliverable bond.

        Net Basis = Gross Basis - Carry

        Where carry includes:
        - Financing cost (repo)
        - Coupon income

        Args:
            bond_index: Index of bond in deliverable basket
            bond_dirty_price: Current dirty price of the bond
            carry: Net carry (coupon income - financing cost)

        Returns:
            Net basis
        """
        gross_basis = self.calculate_gross_basis(bond_index, bond_dirty_price)
        return gross_basis - carry

    def calculate_implied_repo_rate(
        self, bond_index: int, bond_dirty_price: float, valuation_date: datetime
    ) -> float:
        """
        Calculate implied repo rate for a deliverable bond.

        The implied repo rate is the financing rate that makes the net basis
        equal to zero. Higher implied repo = more attractive for delivery.

        Args:
            bond_index: Index of bond in deliverable basket
            bond_dirty_price: Current dirty price of the bond
            valuation_date: Current valuation date

        Returns:
            Implied repo rate (annualized)
        """
        if self.futures_price is None:
            raise ValidationError(
                "Futures price must be set for implied repo calculation"
            )

        if bond_index < 0 or bond_index >= len(self.deliverable_basket):
            raise ValidationError(f"Bond index {bond_index} out of range")

        db = self.deliverable_basket[bond_index]
        bond = db.bond
        cf = self.get_conversion_factor(bond_index)

        time_to_delivery = self.get_time_to_delivery(valuation_date)

        if time_to_delivery <= 0:
            return 0.0

        # Invoice amount at delivery
        accrued_at_delivery = bond.calculate_accrued_interest(self.delivery_date)
        invoice = self.futures_price * cf + accrued_at_delivery

        # Get coupons between now and delivery
        all_cashflows = bond.get_all_cashflows()
        total_coupon = 0.0

        for cf_item in all_cashflows:
            if valuation_date < cf_item.payment_date <= self.delivery_date:
                # Exclude principal repayment
                coupon_amount = cf_item.amount
                if cf_item.payment_date == bond.maturity_date:
                    coupon_amount = cf_item.amount - bond.notional
                if coupon_amount > 0:
                    total_coupon += coupon_amount

        # Implied repo: solve for r in
        # bond_dirty * exp(r*T) = invoice + FV(coupons)
        # Approximate: assume coupons received at midpoint
        adjusted_invoice = invoice + total_coupon

        if bond_dirty_price <= 0:
            return 0.0

        implied_repo = math.log(adjusted_invoice / bond_dirty_price) / time_to_delivery

        return implied_repo

    def find_ctd_bond(
        self, bond_prices: List[float], valuation_date: datetime
    ) -> Tuple[int, float]:
        """
        Find the Cheapest-to-Deliver bond.

        The CTD bond is the one with the highest implied repo rate
        (equivalently, the lowest net basis or delivery cost).

        Args:
            bond_prices: List of dirty prices for each deliverable bond
            valuation_date: Current valuation date

        Returns:
            Tuple of (CTD bond index, implied repo rate)

        Raises:
            ValidationError: If inputs are invalid
        """
        if len(bond_prices) != len(self.deliverable_basket):
            raise ValidationError(
                f"Number of prices ({len(bond_prices)}) must match "
                f"basket size ({len(self.deliverable_basket)})"
            )

        if self.futures_price is None:
            raise ValidationError("Futures price must be set for CTD calculation")

        best_index = -1
        best_repo = float("-inf")

        for i, price in enumerate(bond_prices):
            implied_repo = self.calculate_implied_repo_rate(i, price, valuation_date)

            if implied_repo > best_repo:
                best_repo = implied_repo
                best_index = i

        if best_index < 0:
            raise ValidationError("Could not find CTD bond")

        return best_index, best_repo

    def calculate_theoretical_futures_price(
        self,
        ctd_bond_index: int,
        ctd_dirty_price: float,
        valuation_date: datetime,
        repo_rate: float,
    ) -> float:
        """
        Calculate theoretical futures price from CTD bond.

        Futures Price = (Bond Forward Price - Accrued at Delivery) / CF

        Where:
            Bond Forward Price = Dirty Price * exp(r*T) - FV(coupons)

        Args:
            ctd_bond_index: Index of CTD bond in basket
            ctd_dirty_price: Current dirty price of CTD bond
            valuation_date: Current valuation date
            repo_rate: Financing rate

        Returns:
            Theoretical futures price
        """
        if ctd_bond_index < 0 or ctd_bond_index >= len(self.deliverable_basket):
            raise ValidationError(f"Invalid CTD bond index: {ctd_bond_index}")

        db = self.deliverable_basket[ctd_bond_index]
        bond = db.bond
        cf = self.get_conversion_factor(ctd_bond_index)

        time_to_delivery = self.get_time_to_delivery(valuation_date)

        if time_to_delivery <= 0:
            # At delivery, futures = spot adjusted
            accrued = bond.calculate_accrued_interest(self.delivery_date)
            return (ctd_dirty_price - accrued) / cf

        # Forward dirty price
        carry_factor = math.exp(repo_rate * time_to_delivery)
        forward_dirty = ctd_dirty_price * carry_factor

        # Subtract future value of coupons
        all_cashflows = bond.get_all_cashflows()
        fv_coupons = 0.0

        for cf_item in all_cashflows:
            if valuation_date < cf_item.payment_date <= self.delivery_date:
                coupon_amount = cf_item.amount
                if cf_item.payment_date == bond.maturity_date:
                    coupon_amount = cf_item.amount - bond.notional
                if coupon_amount > 0:
                    time_to_delivery_from_coupon = (
                        self.delivery_date - cf_item.payment_date
                    ).days / 365.0
                    fv_coupons += coupon_amount * math.exp(
                        repo_rate * time_to_delivery_from_coupon
                    )

        forward_dirty -= fv_coupons

        # Subtract accrued at delivery
        accrued_at_delivery = bond.calculate_accrued_interest(self.delivery_date)
        forward_clean = forward_dirty - accrued_at_delivery

        # Adjust by conversion factor
        theoretical_futures = forward_clean / cf

        return theoretical_futures

    def calculate_invoice_price(self, bond_index: int) -> float:
        """
        Calculate invoice price for delivering a specific bond.

        Invoice = Futures Price × CF + Accrued at Delivery

        Args:
            bond_index: Index of bond being delivered

        Returns:
            Invoice price

        Raises:
            ValidationError: If futures price not set
        """
        if self.futures_price is None:
            raise ValidationError("Futures price must be set for invoice calculation")

        if bond_index < 0 or bond_index >= len(self.deliverable_basket):
            raise ValidationError(f"Invalid bond index: {bond_index}")

        cf = self.get_conversion_factor(bond_index)
        bond = self.deliverable_basket[bond_index].bond
        accrued = bond.calculate_accrued_interest(self.delivery_date)

        return self.futures_price * cf + accrued

    def get_basket_size(self) -> int:
        """Get the number of bonds in the deliverable basket."""
        return len(self.deliverable_basket)

    def get_bond(self, index: int) -> FixedBond:
        """
        Get a bond from the deliverable basket.

        Args:
            index: Index in the basket

        Returns:
            The FixedBond at the specified index
        """
        if index < 0 or index >= len(self.deliverable_basket):
            raise ValidationError(
                f"Bond index {index} out of range [0, {len(self.deliverable_basket)})"
            )
        return self.deliverable_basket[index].bond

    def __repr__(self):
        price_str = f", price={self.futures_price:.4f}" if self.futures_price else ""
        return (
            f"BondFutures(delivery={self.delivery_date.date()}, "
            f"basket_size={len(self.deliverable_basket)}, "
            f"size={self.contract_size:,.0f}{price_str})"
        )
