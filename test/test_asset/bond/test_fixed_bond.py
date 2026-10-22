"""
Comprehensive tests for fixed bond implementation.
"""
import unittest
from datetime import datetime
import math

from quantark.asset.bond.product.couponbond.fixed_bond import FixedBond, create_simple_fixed_bond
from quantark.asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from quantark.asset.bond.schedule.cashflow import calculate_accrued_interest
from quantark.param.rrf.rate_curve import FlatRateCurve, LinearRateCurve, LogLinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import (
    DayCountConvention,
    BusinessDayConvention,
    CalendarType,
    create_calendar,
    calculate_day_count_fraction
)
from quantark.util.enum import PaymentFrequency
from quantark.util.exceptions import ValidationError


class TestDayCountConventions(unittest.TestCase):
    """Test day count convention calculations."""
    
    def test_act_360(self):
        """Test ACT/360 calculation."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 7, 1)  # 182 days
        
        fraction = calculate_day_count_fraction(
            start, end, DayCountConvention.ACT_360
        )
        
        expected = 182 / 360
        self.assertAlmostEqual(fraction, expected, places=10)
    
    def test_act_365(self):
        """Test ACT/365 calculation."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 7, 1)  # 182 days
        
        fraction = calculate_day_count_fraction(
            start, end, DayCountConvention.ACT_365
        )
        
        expected = 182 / 365
        self.assertAlmostEqual(fraction, expected, places=10)
    
    def test_thirty_360_us(self):
        """Test 30/360 US calculation."""
        start = datetime(2024, 1, 31)
        end = datetime(2024, 2, 29)
        
        fraction = calculate_day_count_fraction(
            start, end, DayCountConvention.THIRTY_360_US
        )
        
        # 31 -> 30, so difference is 30 days
        expected = 30 / 360
        self.assertAlmostEqual(fraction, expected, places=10)


class TestBusinessCalendar(unittest.TestCase):
    """Test business day calendar."""
    
    def test_weekend_detection(self):
        """Test weekend detection."""
        calendar = create_calendar(CalendarType.NONE)
        
        # Saturday
        saturday = datetime(2024, 1, 6)
        self.assertFalse(calendar.is_business_day(saturday))
        
        # Monday
        monday = datetime(2024, 1, 8)
        self.assertTrue(calendar.is_business_day(monday))
    
    def test_following_convention(self):
        """Test following business day convention."""
        calendar = create_calendar(CalendarType.NONE)
        
        # Saturday -> Monday
        saturday = datetime(2024, 1, 6)
        adjusted = calendar.adjust_date(saturday, BusinessDayConvention.FOLLOWING)
        
        self.assertEqual(adjusted.day, 8)  # Monday
        self.assertTrue(calendar.is_business_day(adjusted))
    
    def test_us_holidays(self):
        """Test US holiday calendar."""
        calendar = create_calendar(CalendarType.US, year_range=(2024, 2024))

        # New Year's Day 2024
        new_year = datetime(2024, 1, 1)
        self.assertTrue(calendar.is_holiday(new_year))
        self.assertFalse(calendar.is_business_day(new_year))

    def test_china_sse_holiday_file(self):
        """Test China SSE holiday file resolution."""
        calendar = create_calendar(CalendarType.CHINA_SSE, year_range=(2020, 2020))

        new_year = datetime(2020, 1, 1)
        self.assertTrue(calendar.is_holiday(new_year))

    def test_china_sse_fallback_to_national(self):
        """Test fallback to national calendar when exchange CSV is missing."""
        calendar = create_calendar(CalendarType.CHINA_SSE, year_range=(1900, 1900))

        new_year = datetime(1900, 1, 1)
        self.assertTrue(calendar.is_holiday(new_year))


class TestRateCurveInterpolation(unittest.TestCase):
    """Test rate curve interpolation methods."""
    
    def test_flat_curve(self):
        """Test flat rate curve."""
        curve = FlatRateCurve(rate=0.05)
        
        # Rate should be constant
        self.assertEqual(curve.get_rate(1.0), 0.05)
        self.assertEqual(curve.get_rate(5.0), 0.05)
        
        # Discount factor
        df_1y = curve.get_discount_factor(1.0)
        expected = math.exp(-0.05 * 1.0)
        self.assertAlmostEqual(df_1y, expected, places=10)
    
    def test_linear_interpolation(self):
        """Test linear rate interpolation."""
        pillars = [(1.0, 0.03), (2.0, 0.05)]
        curve = LinearRateCurve(pillars)
        
        # Midpoint should be 4%
        rate = curve.get_rate(1.5)
        self.assertAlmostEqual(rate, 0.04, places=10)
        
        # Endpoints
        self.assertAlmostEqual(curve.get_rate(1.0), 0.03, places=10)
        self.assertAlmostEqual(curve.get_rate(2.0), 0.05, places=10)
    
    def test_log_linear_interpolation(self):
        """Test log-linear interpolation."""
        pillars = [(1.0, 0.03), (2.0, 0.05)]
        curve = LogLinearRateCurve(pillars)
        
        # Should produce smooth discount factors
        df_1 = curve.get_discount_factor(1.0)
        df_2 = curve.get_discount_factor(2.0)
        
        self.assertAlmostEqual(df_1, math.exp(-0.03 * 1.0), places=10)
        self.assertAlmostEqual(df_2, math.exp(-0.05 * 2.0), places=10)


class TestFixedBond(unittest.TestCase):
    """Test fixed bond product."""
    
    def test_simple_bond_creation(self):
        """Test creating a simple bond."""
        bond = create_simple_fixed_bond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            denominator=1000.0,
            coupon_rate=0.05,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL
        )
        
        self.assertEqual(bond.denominator, 1000.0)
        self.assertEqual(bond.coupon_rate, 0.05)
        self.assertEqual(bond.payment_frequency, PaymentFrequency.SEMI_ANNUAL)
    
    def test_cashflow_generation(self):
        """Test cashflow generation."""
        bond = create_simple_fixed_bond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2026, 1, 1),  # 2 years
            denominator=1000.0,
            coupon_rate=0.06,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL
        )
        
        cashflows = bond.get_all_cashflows()
        
        # Should have 4 coupon payments (2 years * 2 per year)
        self.assertEqual(len(cashflows), 4)
        
        # Last payment should include principal
        last_payment = cashflows[-1].amount
        self.assertGreater(last_payment, 1000.0)  # Principal + coupon
    
    def test_coupon_payment_calculation(self):
        """Test regular coupon payment amount."""
        bond = create_simple_fixed_bond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            denominator=1000.0,
            coupon_rate=0.06,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL
        )
        
        # Semi-annual payment should be 6% / 2 = 3%
        expected = 1000.0 * 0.06 / 2
        self.assertEqual(bond.get_coupon_payment(), expected)
    
    def test_accrued_interest(self):
        """Test accrued interest calculation."""
        bond = create_simple_fixed_bond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            denominator=1000.0,
            coupon_rate=0.06,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL
        )
        
        # Accrued interest halfway through first period
        settlement_date = datetime(2024, 4, 1)  # ~90 days
        accrued = bond.calculate_accrued_interest(settlement_date)
        
        # Should be roughly half a coupon payment
        expected_range = (10.0, 20.0)  # Rough estimate
        self.assertTrue(expected_range[0] < accrued < expected_range[1])


class TestBondPricing(unittest.TestCase):
    """Test bond pricing engine."""
    
    def test_par_bond_pricing(self):
        """Test that bond with coupon = yield prices at par."""
        bond = create_simple_fixed_bond(
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2028, 1, 1),
            denominator=1000.0,
            coupon_rate=0.05,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL
        )
        
        # Price with same yield as coupon
        valuation_date = datetime(2024, 1, 1)
        rate_curve = FlatRateCurve(rate=0.05)
        pricing_env = PricingEnvironment(
            rate_curve=rate_curve,
            valuation_date=valuation_date
        )
        
        engine = BondDiscountEngine(pricing_env)
        price = engine.clean_price(bond, valuation_date, valuation_date)
        
        # Should be close to par (1000)
        self.assertAlmostEqual(price, 1000.0, delta=5.0)
    
    def test_premium_bond_pricing(self):
        """Test that bond with coupon > yield prices at premium."""
        bond = create_simple_fixed_bond(
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2028, 1, 1),
            denominator=1000.0,
            coupon_rate=0.06,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL
        )
        
        # Price with lower yield
        valuation_date = datetime(2024, 1, 1)
        rate_curve = FlatRateCurve(rate=0.04)
        pricing_env = PricingEnvironment(
            rate_curve=rate_curve,
            valuation_date=valuation_date
        )
        
        engine = BondDiscountEngine(pricing_env)
        price = engine.clean_price(bond, valuation_date, valuation_date)
        
        # Should be above par
        self.assertGreater(price, 1000.0)
    
    def test_discount_bond_pricing(self):
        """Test that bond with coupon < yield prices at discount."""
        bond = create_simple_fixed_bond(
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2028, 1, 1),
            denominator=1000.0,
            coupon_rate=0.04,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL
        )
        
        # Price with higher yield
        valuation_date = datetime(2024, 1, 1)
        rate_curve = FlatRateCurve(rate=0.06)
        pricing_env = PricingEnvironment(
            rate_curve=rate_curve,
            valuation_date=valuation_date
        )
        
        engine = BondDiscountEngine(pricing_env)
        price = engine.clean_price(bond, valuation_date, valuation_date)
        
        # Should be below par
        self.assertLess(price, 1000.0)
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        bond = create_simple_fixed_bond(
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2028, 1, 1),
            denominator=1000.0,
            coupon_rate=0.05,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL
        )
        
        valuation_date = datetime(2024, 1, 1)
        rate_curve = FlatRateCurve(rate=0.05)
        pricing_env = PricingEnvironment(
            rate_curve=rate_curve,
            valuation_date=valuation_date
        )
        
        engine = BondDiscountEngine(pricing_env)
        duration = engine.modified_duration(bond)
        
        # Duration should be positive and less than time to maturity
        self.assertGreater(duration, 0)
        self.assertLess(duration, 5.0)  # Less than 5 years
    
    def test_yield_to_maturity(self):
        """Test YTM calculation."""
        bond = create_simple_fixed_bond(
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2028, 1, 1),
            denominator=1000.0,
            coupon_rate=0.05,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL
        )
        
        valuation_date = datetime(2024, 1, 1)
        rate_curve = FlatRateCurve(rate=0.05)
        pricing_env = PricingEnvironment(
            rate_curve=rate_curve,
            valuation_date=valuation_date
        )
        
        engine = BondDiscountEngine(pricing_env)
        price = engine.clean_price(bond, valuation_date, valuation_date)
        
        # Calculate YTM
        ytm = engine.yield_to_maturity(bond, price, valuation_date, valuation_date)
        
        # YTM should be close to the rate curve (5%)
        self.assertAlmostEqual(ytm, 0.05, delta=0.005)


class TestValidation(unittest.TestCase):
    """Test validation and error handling."""
    
    def test_invalid_dates(self):
        """Test that invalid dates raise errors."""
        with self.assertRaises(ValidationError):
            create_simple_fixed_bond(
                issue_date=datetime(2028, 1, 1),
                maturity_date=datetime(2024, 1, 1),  # Before issue
                denominator=1000.0,
                coupon_rate=0.05
            )
    
    def test_negative_denominator(self):
        """Test that negative denominator raises error."""
        with self.assertRaises(ValidationError):
            create_simple_fixed_bond(
                issue_date=datetime(2024, 1, 1),
                maturity_date=datetime(2028, 1, 1),
                denominator=-1000.0,  # Negative
                coupon_rate=0.05
            )
    
    def test_negative_coupon(self):
        """Test that negative coupon raises error."""
        with self.assertRaises(ValidationError):
            create_simple_fixed_bond(
                issue_date=datetime(2024, 1, 1),
                maturity_date=datetime(2028, 1, 1),
                denominator=1000.0,
                coupon_rate=-0.05  # Negative
            )


if __name__ == '__main__':
    unittest.main()
