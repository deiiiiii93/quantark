"""
Discount-based pricing engine for bonds.
"""
from datetime import datetime
from typing import Optional
import math

from asset.bond.product.base_bond_product import BaseBondProduct
from priceenv import PricingEnvironment
from util.exceptions import ValidationError, MarketDataError


class BondDiscountEngine:
    """
    Discount-based pricing engine for bonds.
    
    This engine prices bonds by discounting future cashflows using
    the risk-free rate curve from the pricing environment.
    
    Supports:
    - Clean price (without accrued interest)
    - Dirty price (with accrued interest)
    - Yield to maturity calculation
    - Duration and convexity
    """
    
    def __init__(self, pricing_env: PricingEnvironment):
        """
        Initialize bond discount engine.
        
        Args:
            pricing_env: Pricing environment with rate curve
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")
        
        if pricing_env.rate_curve is None:
            raise MarketDataError("Rate curve is required for bond pricing")
        
        self.pricing_env = pricing_env
    
    def price(
        self,
        bond: BaseBondProduct,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate bond dirty price (present value including accrued interest).
        
        Args:
            bond: Bond product to price
            valuation_date: Date to value the bond (default: pricing env valuation date)
            settlement_date: Settlement date for trade (default: valuation_date)
            
        Returns:
            Dirty price (present value of all future cashflows)
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date
        
        if settlement_date is None:
            settlement_date = valuation_date
        
        # Check if bond has matured
        if bond.is_expired(valuation_date):
            return 0.0
        
        # Get future cashflows
        cashflows = bond.get_cashflows(settlement_date)
        
        if not cashflows:
            return 0.0
        
        # Discount each cashflow
        pv = 0.0
        for cf in cashflows:
            # Calculate time to payment
            time_to_payment = (cf.payment_date - valuation_date).days / 365.0
            
            if time_to_payment < 0:
                continue  # Skip past cashflows
            
            # Get discount factor from rate curve
            discount_factor = self.pricing_env.get_discount_factor(time_to_payment)
            
            # Add discounted cashflow to present value
            pv += cf.amount * discount_factor
        
        return pv
    
    def dirty_price(
        self,
        bond: BaseBondProduct,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate bond dirty price (alias for price method).
        
        Args:
            bond: Bond product to price
            valuation_date: Date to value the bond
            settlement_date: Settlement date for trade
            
        Returns:
            Dirty price
        """
        return self.price(bond, valuation_date, settlement_date)
    
    def clean_price(
        self,
        bond: BaseBondProduct,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate bond clean price (dirty price - accrued interest).
        
        Args:
            bond: Bond product to price
            valuation_date: Date to value the bond
            settlement_date: Settlement date for trade
            
        Returns:
            Clean price
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date
        
        if settlement_date is None:
            settlement_date = valuation_date
        
        # Get dirty price
        dirty = self.dirty_price(bond, valuation_date, settlement_date)
        
        # Calculate accrued interest
        accrued = bond.calculate_accrued_interest(settlement_date)
        
        # Clean price = dirty price - accrued interest
        return dirty - accrued
    
    def accrued_interest(
        self,
        bond: BaseBondProduct,
        settlement_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate accrued interest.
        
        Args:
            bond: Bond product
            settlement_date: Settlement date
            
        Returns:
            Accrued interest amount
        """
        if settlement_date is None:
            settlement_date = self.pricing_env.valuation_date
        
        return bond.calculate_accrued_interest(settlement_date)
    
    def yield_to_maturity(
        self,
        bond: BaseBondProduct,
        price: float,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        clean_price: bool = True,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """
        Calculate yield to maturity using Newton-Raphson iteration.
        
        Args:
            bond: Bond product
            price: Market price (clean or dirty based on clean_price flag)
            valuation_date: Valuation date
            settlement_date: Settlement date
            clean_price: Whether price is clean price (default: True)
            max_iterations: Maximum iterations for solver
            tolerance: Convergence tolerance
            
        Returns:
            Yield to maturity (annualized)
            
        Raises:
            ValidationError: If convergence fails
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date
        
        if settlement_date is None:
            settlement_date = valuation_date
        
        if price <= 0:
            raise ValidationError(f"Price must be positive, got {price}")
        
        # Convert clean price to dirty price if needed
        if clean_price:
            accrued = bond.calculate_accrued_interest(settlement_date)
            target_price = price + accrued
        else:
            target_price = price
        
        # Get cashflows
        cashflows = bond.get_cashflows(settlement_date)
        
        if not cashflows:
            raise ValidationError("No future cashflows to calculate yield")
        
        # Initial guess: use coupon rate or estimate from price
        if hasattr(bond, 'coupon_rate'):
            ytm = bond.coupon_rate
        else:
            # Rough estimate from price
            ttm = bond.time_to_maturity(valuation_date)
            ytm = -math.log(price / bond.get_notional()) / ttm if ttm > 0 else 0.05
        
        # Newton-Raphson iteration
        for iteration in range(max_iterations):
            # Calculate price and duration at current yield
            pv = 0.0
            duration = 0.0
            
            for cf in cashflows:
                time_to_payment = (cf.payment_date - valuation_date).days / 365.0
                
                if time_to_payment < 0:
                    continue
                
                df = math.exp(-ytm * time_to_payment)
                pv += cf.amount * df
                duration += cf.amount * time_to_payment * df
            
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
            if ytm < -0.5 or ytm > 1.0:
                ytm = max(-0.5, min(1.0, ytm))
        
        raise ValidationError(
            f"Yield to maturity did not converge after {max_iterations} iterations"
        )
    
    def modified_duration(
        self,
        bond: BaseBondProduct,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate modified duration.
        
        Modified duration measures the price sensitivity to yield changes.
        
        Args:
            bond: Bond product
            valuation_date: Valuation date
            settlement_date: Settlement date
            
        Returns:
            Modified duration
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date
        
        if settlement_date is None:
            settlement_date = valuation_date
        
        # Get price and cashflows
        base_price = self.price(bond, valuation_date, settlement_date)
        
        if base_price == 0:
            return 0.0
        
        cashflows = bond.get_cashflows(settlement_date)
        
        if not cashflows:
            return 0.0
        
        # Calculate weighted average time to cashflows
        weighted_time = 0.0
        
        for cf in cashflows:
            time_to_payment = (cf.payment_date - valuation_date).days / 365.0
            
            if time_to_payment < 0:
                continue
            
            discount_factor = self.pricing_env.get_discount_factor(time_to_payment)
            pv = cf.amount * discount_factor
            
            weighted_time += pv * time_to_payment
        
        # Modified duration = weighted average time / price
        return weighted_time / base_price
    
    def macaulay_duration(
        self,
        bond: BaseBondProduct,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate Macaulay duration.
        
        Args:
            bond: Bond product
            valuation_date: Valuation date
            settlement_date: Settlement date
            
        Returns:
            Macaulay duration (in years)
        """
        # For continuously compounded yields, Macaulay = Modified duration
        return self.modified_duration(bond, valuation_date, settlement_date)
    
    def convexity(
        self,
        bond: BaseBondProduct,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate convexity.
        
        Convexity measures the curvature of the price-yield relationship.
        
        Args:
            bond: Bond product
            valuation_date: Valuation date
            settlement_date: Settlement date
            
        Returns:
            Convexity
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date
        
        if settlement_date is None:
            settlement_date = valuation_date
        
        # Get price and cashflows
        base_price = self.price(bond, valuation_date, settlement_date)
        
        if base_price == 0:
            return 0.0
        
        cashflows = bond.get_cashflows(settlement_date)
        
        if not cashflows:
            return 0.0
        
        # Calculate weighted average time squared
        weighted_time_sq = 0.0
        
        for cf in cashflows:
            time_to_payment = (cf.payment_date - valuation_date).days / 365.0
            
            if time_to_payment < 0:
                continue
            
            discount_factor = self.pricing_env.get_discount_factor(time_to_payment)
            pv = cf.amount * discount_factor
            
            weighted_time_sq += pv * time_to_payment * time_to_payment
        
        # Convexity
        return weighted_time_sq / base_price
    
    def dv01(
        self,
        bond: BaseBondProduct,
        valuation_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate DV01 (dollar value of one basis point).
        
        DV01 measures the change in bond price for a 1 basis point
        change in yield.
        
        Args:
            bond: Bond product
            valuation_date: Valuation date
            settlement_date: Settlement date
            
        Returns:
            DV01 (price change per basis point)
        """
        mod_dur = self.modified_duration(bond, valuation_date, settlement_date)
        price = self.price(bond, valuation_date, settlement_date)
        
        # DV01 = Modified Duration * Price * 0.0001
        return mod_dur * price * 0.0001
    
    def __repr__(self):
        return f"BondDiscountEngine(valuation_date={self.pricing_env.valuation_date.date()})"

