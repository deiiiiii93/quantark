"""
Pricing engine for bond futures contracts with CTD analysis.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from asset.bond.product.futures.bond_futures import BondFutures, DeliverableBond
from asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from priceenv import PricingEnvironment
from util.exceptions import ValidationError, PricingError


@dataclass
class BondAnalysis:
    """
    Analysis results for a single deliverable bond.

    Attributes:
        bond_index: Index in the deliverable basket
        dirty_price: Current dirty price
        clean_price: Current clean price
        accrued_interest: Current accrued interest
        conversion_factor: Exchange conversion factor
        gross_basis: Bond Price - Futures × CF
        net_basis: Gross Basis - Carry
        implied_repo_rate: Implied financing rate
        invoice_price: Invoice at delivery
        is_ctd: Whether this is the CTD bond
    """

    bond_index: int
    dirty_price: float
    clean_price: float
    accrued_interest: float
    conversion_factor: float
    gross_basis: float
    net_basis: float
    implied_repo_rate: float
    invoice_price: float
    is_ctd: bool = False


@dataclass
class BondFuturesResults:
    """
    Results from bond futures pricing.

    Attributes:
        theoretical_futures_price: Theoretical price from CTD bond
        ctd_bond_index: Index of CTD bond in basket
        ctd_implied_repo: Implied repo rate of CTD
        bond_analyses: Analysis for each deliverable bond
        dv01: Dollar value of 1bp rate change
        modified_duration: CTD-adjusted duration
        convexity: CTD-adjusted convexity
        time_to_delivery: Time to delivery in years
    """

    theoretical_futures_price: float
    ctd_bond_index: int
    ctd_implied_repo: float
    bond_analyses: List[BondAnalysis]
    dv01: float
    modified_duration: float
    convexity: float
    time_to_delivery: float


class BondFuturesEngine:
    """
    Pricing engine for bond futures with CTD (Cheapest to Deliver) logic.

    This engine:
    1. Prices each bond in the deliverable basket
    2. Calculates conversion factors and basis for each bond
    3. Identifies the CTD bond (highest implied repo rate)
    4. Prices the futures from the CTD bond
    5. Calculates CTD-adjusted risk measures

    The CTD bond is the one that a rational deliverer would choose to
    deliver, as it minimizes their cost (maximizes profit) at delivery.

    Attributes:
        pricing_env: Pricing environment with market data
        bond_engine: Underlying bond pricing engine
        bump_size: Basis point bump for Greeks (default: 1bp)
        repo_rate: Financing rate for carry calculations
    """

    def __init__(
        self,
        pricing_env: PricingEnvironment,
        repo_rate: float = 0.05,
        bump_size: float = 0.0001,
    ):
        """
        Initialize bond futures engine.

        Args:
            pricing_env: Pricing environment with rate curve
            repo_rate: Financing rate for carry calculations (default: 5%)
            bump_size: Bump size for finite difference Greeks (default: 1bp)
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")

        self.pricing_env = pricing_env
        self.bond_engine = BondDiscountEngine(pricing_env)
        self.repo_rate = repo_rate
        self.bump_size = bump_size

    def price(
        self,
        futures: BondFutures,
        valuation_date: Optional[datetime] = None,
        calculate_greeks: bool = True,
    ) -> BondFuturesResults:
        """
        Price a bond futures contract and identify CTD.

        Args:
            futures: Bond futures contract to price
            valuation_date: Valuation date (default: pricing env date)
            calculate_greeks: Whether to calculate Greeks (default: True)

        Returns:
            BondFuturesResults with pricing and CTD analysis

        Raises:
            PricingError: If pricing fails
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        # Check if expired
        if futures.is_expired(valuation_date):
            raise PricingError("Futures contract has expired")

        time_to_delivery = futures.get_time_to_delivery(valuation_date)

        # Price each deliverable bond
        bond_prices = []
        bond_analyses = []

        for i, db in enumerate(futures.deliverable_basket):
            bond = db.bond

            # Price the bond
            dirty_price = self.bond_engine.dirty_price(
                bond, valuation_date, valuation_date
            )
            clean_price = self.bond_engine.clean_price(
                bond, valuation_date, valuation_date
            )
            accrued = bond.calculate_accrued_interest(valuation_date)

            bond_prices.append(dirty_price)

            cf = futures.get_conversion_factor(i)

            # Calculate basis and implied repo (if futures price set)
            if futures.futures_price is not None:
                gross_basis = futures.calculate_gross_basis(i, dirty_price)
                implied_repo = futures.calculate_implied_repo_rate(
                    i, dirty_price, valuation_date
                )
                invoice = futures.calculate_invoice_price(i)

                # Calculate carry for net basis
                carry = self._calculate_carry(
                    bond, dirty_price, valuation_date, futures.delivery_date
                )
                net_basis = gross_basis - carry
            else:
                gross_basis = 0.0
                net_basis = 0.0
                implied_repo = 0.0
                invoice = 0.0

            analysis = BondAnalysis(
                bond_index=i,
                dirty_price=dirty_price,
                clean_price=clean_price,
                accrued_interest=accrued,
                conversion_factor=cf,
                gross_basis=gross_basis,
                net_basis=net_basis,
                implied_repo_rate=implied_repo,
                invoice_price=invoice,
                is_ctd=False,
            )
            bond_analyses.append(analysis)

        # Find CTD bond
        if futures.futures_price is not None:
            ctd_index, ctd_repo = futures.find_ctd_bond(bond_prices, valuation_date)
        else:
            # If no futures price, use first bond as pseudo-CTD
            ctd_index = 0
            ctd_repo = 0.0

        # Mark CTD
        bond_analyses[ctd_index].is_ctd = True

        # Calculate theoretical futures price from CTD
        theoretical_price = futures.calculate_theoretical_futures_price(
            ctd_index, bond_prices[ctd_index], valuation_date, self.repo_rate
        )

        # Calculate Greeks (avoid recursion by skipping in bumped engines)
        if calculate_greeks:
            dv01 = self._calculate_futures_dv01(
                futures, bond_prices, ctd_index, valuation_date, theoretical_price
            )
            mod_dur = self._calculate_futures_duration(theoretical_price, dv01)
            convexity = self._calculate_futures_convexity(
                futures, bond_prices, ctd_index, valuation_date, theoretical_price
            )
        else:
            dv01 = 0.0
            mod_dur = 0.0
            convexity = 0.0

        return BondFuturesResults(
            theoretical_futures_price=theoretical_price,
            ctd_bond_index=ctd_index,
            ctd_implied_repo=ctd_repo,
            bond_analyses=bond_analyses,
            dv01=dv01,
            modified_duration=mod_dur,
            convexity=convexity,
            time_to_delivery=time_to_delivery,
        )

    def _calculate_carry(
        self,
        bond,
        dirty_price: float,
        valuation_date: datetime,
        delivery_date: datetime,
    ) -> float:
        """
        Calculate carry cost between valuation and delivery.

        Carry = Coupon Income - Financing Cost
        """
        time_to_delivery = (delivery_date - valuation_date).days / 365.0

        if time_to_delivery <= 0:
            return 0.0

        # Financing cost
        financing_cost = dirty_price * (math.exp(self.repo_rate * time_to_delivery) - 1)

        # Coupon income
        coupon_income = 0.0
        all_cashflows = bond.get_all_cashflows()

        for cf in all_cashflows:
            if valuation_date < cf.payment_date <= delivery_date:
                # Exclude principal
                amount = cf.amount
                if cf.payment_date == bond.maturity_date:
                    amount = cf.amount - bond.notional
                if amount > 0:
                    coupon_income += amount

        carry = coupon_income - financing_cost
        return carry

    def _calculate_futures_dv01(
        self,
        futures: BondFutures,
        bond_prices: List[float],
        ctd_index: int,
        valuation_date: datetime,
        base_price: float,
    ) -> float:
        """Calculate DV01 of futures (CTD-adjusted)."""
        from param.rrf.rate_curve import FlatRateCurve

        # Get base rate
        base_rate = self.pricing_env.rate_curve.get_rate(1.0)

        # Create bumped environment with a new flat curve
        bumped_rate = base_rate + self.bump_size
        bumped_curve = FlatRateCurve(rate=bumped_rate)

        env_up = PricingEnvironment(
            rate_curve=bumped_curve,
            valuation_date=self.pricing_env.valuation_date,
            spot_quote=self.pricing_env.spot_quote,
            vol_surface=self.pricing_env.vol_surface,
            div_yield=self.pricing_env.div_yield,
        )

        # Create new engine and price without Greeks to avoid recursion
        engine_up = BondFuturesEngine(env_up, self.repo_rate, self.bump_size)
        results_up = engine_up.price(futures, valuation_date, calculate_greeks=False)

        # DV01 = change in futures price for 1bp increase
        dv01 = base_price - results_up.theoretical_futures_price

        # Adjust by conversion factor of CTD
        cf = futures.get_conversion_factor(ctd_index)
        dv01_adjusted = dv01 / cf

        return dv01_adjusted

    def _calculate_futures_duration(self, futures_price: float, dv01: float) -> float:
        """Calculate modified duration of futures (CTD-adjusted)."""
        if futures_price == 0:
            return 0.0

        # Duration = DV01 / (Price * bump_size)
        mod_dur = dv01 / (futures_price * self.bump_size)

        return mod_dur

    def _calculate_futures_convexity(
        self,
        futures: BondFutures,
        bond_prices: List[float],
        ctd_index: int,
        valuation_date: datetime,
        futures_price: float,
    ) -> float:
        """Calculate convexity of futures (CTD-adjusted)."""
        from param.rrf.rate_curve import FlatRateCurve

        if futures_price == 0:
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
        engine_up = BondFuturesEngine(env_up, self.repo_rate, self.bump_size)
        results_up = engine_up.price(futures, valuation_date, calculate_greeks=False)

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
        engine_down = BondFuturesEngine(env_down, self.repo_rate, self.bump_size)
        results_down = engine_down.price(
            futures, valuation_date, calculate_greeks=False
        )

        price_up = results_up.theoretical_futures_price
        price_down = results_down.theoretical_futures_price

        # Convexity = (P_up - 2*P_base + P_down) / (P_base * bump^2)
        convexity = (price_up - 2 * futures_price + price_down) / (
            futures_price * self.bump_size * self.bump_size
        )

        return convexity

    def calculate_greeks(
        self, futures: BondFutures, valuation_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate Greeks for bond futures.

        Greeks calculated:
        - theoretical_price: Theoretical futures price from CTD
        - dv01: Dollar value of 1bp rate change
        - modified_duration: CTD-adjusted duration
        - convexity: CTD-adjusted convexity
        - ctd_bond_index: Index of CTD bond
        - ctd_implied_repo: Implied repo rate of CTD
        - basis_point_value: BPV per contract

        Args:
            futures: Bond futures contract
            valuation_date: Valuation date

        Returns:
            Dictionary of Greeks
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        results = self.price(futures, valuation_date)

        greeks = {
            "theoretical_price": results.theoretical_futures_price,
            "dv01": results.dv01,
            "modified_duration": results.modified_duration,
            "convexity": results.convexity,
            "ctd_bond_index": results.ctd_bond_index,
            "ctd_implied_repo": results.ctd_implied_repo,
            "time_to_delivery": results.time_to_delivery,
        }

        # Basis point value per contract
        greeks["basis_point_value"] = results.dv01 * futures.contract_size / 100.0

        return greeks

    def analyze_basis(
        self, futures: BondFutures, valuation_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Analyze basis for all deliverable bonds.

        Returns detailed basis analysis including:
        - Gross basis
        - Net basis
        - Implied repo rate
        - Delivery option value

        Args:
            futures: Bond futures contract
            valuation_date: Valuation date

        Returns:
            List of dictionaries with basis analysis per bond
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        if futures.futures_price is None:
            raise PricingError("Futures price must be set for basis analysis")

        results = self.price(futures, valuation_date)

        analyses = []
        for ba in results.bond_analyses:
            bond = futures.deliverable_basket[ba.bond_index].bond

            analysis = {
                "bond_index": ba.bond_index,
                "bond_description": f"{bond.coupon_rate:.2%} {bond.maturity_date.date()}",
                "dirty_price": ba.dirty_price,
                "clean_price": ba.clean_price,
                "conversion_factor": ba.conversion_factor,
                "gross_basis": ba.gross_basis,
                "net_basis": ba.net_basis,
                "implied_repo_rate": ba.implied_repo_rate,
                "invoice_price": ba.invoice_price,
                "is_ctd": ba.is_ctd,
                "delivery_value": (
                    ba.invoice_price - ba.dirty_price if ba.is_ctd else None
                ),
            }
            analyses.append(analysis)

        # Sort by implied repo (highest first = CTD)
        analyses.sort(key=lambda x: x["implied_repo_rate"], reverse=True)

        return analyses

    def calculate_hedge_ratio(
        self,
        futures: BondFutures,
        target_bond_dv01: float,
        valuation_date: Optional[datetime] = None,
    ) -> float:
        """
        Calculate hedge ratio for hedging a bond position with futures.

        Hedge Ratio = Target Bond DV01 / Futures DV01

        Args:
            futures: Bond futures contract
            target_bond_dv01: DV01 of the bond position to hedge
            valuation_date: Valuation date

        Returns:
            Number of futures contracts needed (may be fractional)
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        results = self.price(futures, valuation_date)

        if results.dv01 == 0:
            raise PricingError("Futures DV01 is zero, cannot calculate hedge ratio")

        # Futures DV01 per contract
        futures_dv01_per_contract = results.dv01 * futures.contract_size / 100.0

        # Hedge ratio
        hedge_ratio = target_bond_dv01 / futures_dv01_per_contract

        return hedge_ratio

    def find_delivery_option_value(
        self, futures: BondFutures, valuation_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Estimate the value of the delivery option.

        The delivery option value is approximately the difference between:
        - The theoretical futures price (from CTD)
        - The average price across all deliverables

        A positive value indicates the delivery option is valuable.

        Args:
            futures: Bond futures contract
            valuation_date: Valuation date

        Returns:
            Dictionary with delivery option analysis
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date

        results = self.price(futures, valuation_date)

        # Calculate theoretical price for each bond
        theoretical_prices = []
        for i, ba in enumerate(results.bond_analyses):
            theo_price = futures.calculate_theoretical_futures_price(
                i, ba.dirty_price, valuation_date, self.repo_rate
            )
            theoretical_prices.append(theo_price)

        # CTD price
        ctd_price = results.theoretical_futures_price

        # Average price
        avg_price = (
            sum(theoretical_prices) / len(theoretical_prices)
            if theoretical_prices
            else 0
        )

        # Max price (most expensive to deliver)
        max_price = max(theoretical_prices) if theoretical_prices else 0

        # Delivery option value approximation
        # = what the futures would be worth if we couldn't choose CTD
        option_value = avg_price - ctd_price

        return {
            "ctd_theoretical_price": ctd_price,
            "average_theoretical_price": avg_price,
            "max_theoretical_price": max_price,
            "delivery_option_value": option_value,
            "option_value_pct": option_value / ctd_price * 100 if ctd_price > 0 else 0,
        }

    def __repr__(self):
        return (
            f"BondFuturesEngine(valuation_date={self.pricing_env.valuation_date.date()}, "
            f"repo={self.repo_rate:.2%})"
        )
