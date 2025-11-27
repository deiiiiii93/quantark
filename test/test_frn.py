"""
Comprehensive tests for Floating Rate Note (FRN) implementation.
"""

import unittest
from datetime import datetime
import math

from asset.bond.product.couponbond.frn import (
    FloatingRateBond,
    create_simple_frn,
)
from asset.bond.schedule.cashflow import FloatingCashFlow
from asset.bond.engine.discount.frn_engine import FRNDiscountEngine, FRNPricingResults
from param.index import (
    RateIndex,
    IndexFixing,
    IndexFixingStore,
    SOFR,
    SOFR_3M,
    EURIBOR_3M,
    SHIBOR_3M,
    REPO_7D,
    create_index,
)
from param.rrf import FlatRateCurve
from param.rrf.rate_curve import LinearRateCurve
from priceenv import PricingEnvironment
from util.calendar import (
    DayCountConvention,
    BusinessDayConvention,
    CalendarType,
    create_calendar,
)
from util.enum import PaymentFrequency, ResetConvention
from util.exceptions import ValidationError


class TestRateIndex(unittest.TestCase):
    """Test rate index functionality."""

    def test_predefined_sofr(self):
        """Test SOFR index properties."""
        self.assertEqual(SOFR.name, "SOFR")
        self.assertEqual(SOFR.tenor_months, 0)
        self.assertTrue(SOFR.is_overnight)
        self.assertEqual(SOFR.currency, "USD")
        self.assertEqual(SOFR.day_count_convention, DayCountConvention.ACT_360)

    def test_predefined_sofr_3m(self):
        """Test SOFR 3M index properties."""
        self.assertEqual(SOFR_3M.name, "SOFR_3M")
        self.assertEqual(SOFR_3M.tenor_months, 3)
        self.assertFalse(SOFR_3M.is_overnight)
        self.assertAlmostEqual(SOFR_3M.tenor_years, 0.25, places=4)

    def test_predefined_euribor(self):
        """Test EURIBOR index properties."""
        self.assertEqual(EURIBOR_3M.name, "EURIBOR_3M")
        self.assertEqual(EURIBOR_3M.currency, "EUR")
        self.assertEqual(EURIBOR_3M.calendar_type, CalendarType.TARGET)

    def test_predefined_shibor(self):
        """Test SHIBOR index properties."""
        self.assertEqual(SHIBOR_3M.name, "SHIBOR_3M")
        self.assertEqual(SHIBOR_3M.currency, "CNY")
        self.assertEqual(SHIBOR_3M.calendar_type, CalendarType.CHINA)

    def test_predefined_repo_7d(self):
        """Test REPO 7D (DR007) index properties."""
        self.assertEqual(REPO_7D.name, "REPO_7D")
        self.assertEqual(REPO_7D.currency, "CNY")
        self.assertEqual(REPO_7D.day_count_convention, DayCountConvention.ACT_365)

    def test_create_predefined_index(self):
        """Test creating index from predefined template."""
        sofr = create_index("SOFR")
        self.assertEqual(sofr.name, "SOFR")
        self.assertEqual(sofr.currency, "USD")

    def test_create_custom_index(self):
        """Test creating a custom index."""
        custom = create_index(
            name="CUSTOM_RATE",
            tenor_months=6,
            day_count_convention=DayCountConvention.ACT_365,
            fixing_lag_days=1,
            calendar_type=CalendarType.US,
            currency="USD",
            description="Custom test rate",
        )

        self.assertEqual(custom.name, "CUSTOM_RATE")
        self.assertEqual(custom.tenor_months, 6)
        self.assertEqual(custom.currency, "USD")

    def test_create_custom_index_missing_params(self):
        """Test that custom index requires all parameters."""
        with self.assertRaises(ValidationError):
            create_index(
                name="INCOMPLETE",
                tenor_months=3,
                # Missing other required params
            )


class TestIndexFixingStore(unittest.TestCase):
    """Test index fixing store functionality."""

    def test_add_and_get_fixing(self):
        """Test adding and retrieving fixings."""
        store = IndexFixingStore()

        fixing = IndexFixing(
            fixing_date=datetime(2024, 1, 15), rate=0.0525, index_name="SOFR"
        )
        store.add_fixing(fixing)

        rate = store.get_fixing("SOFR", datetime(2024, 1, 15))
        self.assertAlmostEqual(rate, 0.0525, places=6)

    def test_get_latest_fixing(self):
        """Test getting the latest fixing before a date."""
        store = IndexFixingStore()

        store.add_fixing(IndexFixing(datetime(2024, 1, 10), 0.05, "SOFR"))
        store.add_fixing(IndexFixing(datetime(2024, 1, 15), 0.052, "SOFR"))
        store.add_fixing(IndexFixing(datetime(2024, 1, 20), 0.054, "SOFR"))

        latest = store.get_latest_fixing("SOFR", datetime(2024, 1, 18))

        self.assertIsNotNone(latest)
        self.assertEqual(latest[0], datetime(2024, 1, 15))
        self.assertAlmostEqual(latest[1], 0.052, places=6)

    def test_get_all_fixings_in_range(self):
        """Test getting fixings within a date range."""
        store = IndexFixingStore()

        for day in range(1, 20):
            store.add_fixing(
                IndexFixing(datetime(2024, 1, day), 0.05 + day * 0.001, "SOFR")
            )

        fixings = store.get_all_fixings(
            "SOFR", start_date=datetime(2024, 1, 5), end_date=datetime(2024, 1, 10)
        )

        self.assertEqual(len(fixings), 6)  # Days 5-10


class TestFloatingRateBond(unittest.TestCase):
    """Test FRN product creation and functionality."""

    def test_simple_frn_creation(self):
        """Test creating a simple FRN."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,  # 50bp
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        self.assertEqual(frn.notional, 1000000.0)
        self.assertEqual(frn.spread, 0.0050)
        self.assertEqual(frn.index.name, "SOFR_3M")

    def test_frn_with_cap_floor(self):
        """Test FRN with rate cap and floor."""
        frn = FloatingRateBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
            rate_cap=0.10,  # 10% cap
            rate_floor=0.02,  # 2% floor
        )

        self.assertEqual(frn.rate_cap, 0.10)
        self.assertEqual(frn.rate_floor, 0.02)

    def test_frn_cashflow_generation(self):
        """Test cashflow generation."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2026, 1, 1),  # 2 years
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        cashflows = frn.get_all_floating_cashflows()

        # 2 years * 4 quarters = 8 cashflows
        self.assertEqual(len(cashflows), 8)

        # All should be projected (no historical fixings)
        for cf in cashflows:
            self.assertTrue(cf.is_projected)

    def test_frn_with_historical_fixings(self):
        """Test FRN with historical fixings."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2025, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        # Get the actual fixing date for the first cashflow
        cashflows = frn.get_all_floating_cashflows()
        first_cf = cashflows[0]

        # Add a historical fixing for the actual fixing date
        frn.add_fixing(first_cf.fixing_date, 0.0525)

        # Refresh cashflows
        cashflows = frn.get_all_floating_cashflows()
        first_cf = cashflows[0]

        # First cashflow should now be fixed
        self.assertFalse(first_cf.is_projected)
        self.assertIsNotNone(first_cf.index_fixing)

    def test_frn_in_arrears(self):
        """Test FRN with in-arrears reset."""
        frn = FloatingRateBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2025, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
            reset_convention=ResetConvention.IN_ARREARS,
            lookback_days=5,
        )

        self.assertEqual(frn.reset_convention, ResetConvention.IN_ARREARS)
        self.assertEqual(frn.lookback_days, 5)

    def test_frn_validation_errors(self):
        """Test FRN validation."""
        # Negative notional
        with self.assertRaises(ValidationError):
            create_simple_frn(
                issue_date=datetime(2024, 1, 1),
                maturity_date=datetime(2025, 1, 1),
                notional=-1000000.0,
                index=SOFR_3M,
                spread=0.0050,
            )

        # Maturity before issue
        with self.assertRaises(ValidationError):
            create_simple_frn(
                issue_date=datetime(2025, 1, 1),
                maturity_date=datetime(2024, 1, 1),
                notional=1000000.0,
                index=SOFR_3M,
                spread=0.0050,
            )

    def test_cap_floor_validation(self):
        """Test that cap must be >= floor."""
        with self.assertRaises(ValidationError):
            FloatingRateBond(
                issue_date=datetime(2024, 1, 1),
                maturity_date=datetime(2025, 1, 1),
                notional=1000000.0,
                index=SOFR_3M,
                spread=0.0050,
                payment_frequency=PaymentFrequency.QUARTERLY,
                rate_cap=0.05,
                rate_floor=0.08,  # Floor > Cap
            )


class TestFloatingCashFlow(unittest.TestCase):
    """Test FloatingCashFlow calculations."""

    def test_effective_rate_with_spread(self):
        """Test effective rate calculation."""
        cf = FloatingCashFlow(
            payment_date=datetime(2024, 7, 1),
            accrual_start_date=datetime(2024, 4, 1),
            accrual_end_date=datetime(2024, 7, 1),
            fixing_date=datetime(2024, 4, 1),
            notional=1000000.0,
            spread=0.0050,  # 50bp
            day_count_fraction=0.25,
            index_fixing=0.0525,  # 5.25%
        )

        # Effective rate = index + spread = 5.25% + 0.50% = 5.75%
        expected_rate = 0.0575
        self.assertAlmostEqual(cf.effective_rate, expected_rate, places=6)

    def test_effective_rate_with_cap(self):
        """Test rate cap application."""
        cf = FloatingCashFlow(
            payment_date=datetime(2024, 7, 1),
            accrual_start_date=datetime(2024, 4, 1),
            accrual_end_date=datetime(2024, 7, 1),
            fixing_date=datetime(2024, 4, 1),
            notional=1000000.0,
            spread=0.0050,
            day_count_fraction=0.25,
            index_fixing=0.10,  # 10%
            rate_cap=0.08,  # 8% cap
        )

        # Rate should be capped at 8%
        self.assertAlmostEqual(cf.effective_rate, 0.08, places=6)

    def test_effective_rate_with_floor(self):
        """Test rate floor application."""
        cf = FloatingCashFlow(
            payment_date=datetime(2024, 7, 1),
            accrual_start_date=datetime(2024, 4, 1),
            accrual_end_date=datetime(2024, 7, 1),
            fixing_date=datetime(2024, 4, 1),
            notional=1000000.0,
            spread=0.0050,
            day_count_fraction=0.25,
            index_fixing=0.01,  # 1%
            rate_floor=0.03,  # 3% floor
        )

        # Rate should be floored at 3%
        self.assertAlmostEqual(cf.effective_rate, 0.03, places=6)

    def test_cashflow_amount(self):
        """Test coupon amount calculation."""
        cf = FloatingCashFlow(
            payment_date=datetime(2024, 7, 1),
            accrual_start_date=datetime(2024, 4, 1),
            accrual_end_date=datetime(2024, 7, 1),
            fixing_date=datetime(2024, 4, 1),
            notional=1000000.0,
            spread=0.0050,
            day_count_fraction=0.25,
            index_fixing=0.0525,
        )

        # Amount = notional * rate * dcf = 1M * 5.75% * 0.25 = 14,375
        expected_amount = 1000000.0 * 0.0575 * 0.25
        self.assertAlmostEqual(cf.amount, expected_amount, places=2)


class TestFRNPricing(unittest.TestCase):
    """Test FRN pricing engine."""

    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 1)
        self.rate_curve = FlatRateCurve(rate=0.05)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve, valuation_date=self.valuation_date
        )
        self.engine = FRNDiscountEngine(self.pricing_env)

    def test_par_frn_pricing(self):
        """Test that FRN with zero spread prices near par on reset date."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0,  # Zero spread
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        # Add fixing at current rate
        frn.add_fixing(datetime(2024, 1, 1), 0.05)

        price = self.engine.clean_price(frn)

        # Should be close to par (1,000,000)
        self.assertAlmostEqual(price / frn.notional, 1.0, delta=0.01)

    def test_premium_frn_pricing(self):
        """Test FRN with positive spread prices above par."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0100,  # 100bp above market
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        price = self.engine.clean_price(frn)

        # Should be above par
        self.assertGreater(price, frn.notional)

    def test_discount_frn_pricing(self):
        """Test FRN with negative spread prices below par."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=-0.0050,  # 50bp below market
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        price = self.engine.clean_price(frn)

        # Should be below par
        self.assertLess(price, frn.notional)

    def test_dirty_vs_clean_price(self):
        """Test relationship between dirty and clean price."""
        frn = create_simple_frn(
            issue_date=datetime(2023, 10, 1),  # Started 3 months ago
            maturity_date=datetime(2028, 10, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        # Get the first cashflow fixing date and add fixing
        first_cf = frn.get_all_floating_cashflows()[0]
        frn.add_fixing(first_cf.fixing_date, 0.05)

        # Use a mid-period valuation date
        mid_period_date = datetime(2024, 2, 15)

        dirty = self.engine.dirty_price(frn, mid_period_date, mid_period_date)
        clean = self.engine.clean_price(frn, mid_period_date, mid_period_date)
        accrued = frn.calculate_accrued_interest(mid_period_date)

        # Dirty = Clean + Accrued
        self.assertAlmostEqual(dirty, clean + accrued, places=2)

    def test_accrued_interest(self):
        """Test accrued interest calculation."""
        frn = create_simple_frn(
            issue_date=datetime(2023, 10, 1),
            maturity_date=datetime(2028, 10, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        # Get the first cashflow fixing date
        first_cf = frn.get_all_floating_cashflows()[0]
        frn.add_fixing(first_cf.fixing_date, 0.05)

        # Valuation date mid-period (Feb 15, 2024 - between Jan 1 and Apr 1)
        mid_period_date = datetime(2024, 2, 15)
        accrued = frn.calculate_accrued_interest(mid_period_date)

        # Should be positive (mid-way through period)
        self.assertGreater(accrued, 0)


class TestDiscountMargin(unittest.TestCase):
    """Test Discount Margin calculations."""

    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 1)
        self.rate_curve = FlatRateCurve(rate=0.05)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve, valuation_date=self.valuation_date
        )
        self.engine = FRNDiscountEngine(self.pricing_env)

    def test_discount_margin_at_par(self):
        """Test DM for FRN trading at par equals quoted spread."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,  # 50bp
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        # Price at par
        par_price = frn.notional

        dm = self.engine.discount_margin(frn, par_price)

        # DM should be close to the spread
        self.assertAlmostEqual(dm, frn.spread, delta=0.001)

    def test_discount_margin_premium(self):
        """Test DM for FRN trading at premium."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        # Price above par
        premium_price = frn.notional * 1.02  # 102%

        dm = self.engine.discount_margin(frn, premium_price)

        # DM should be less than spread (trading rich)
        self.assertLess(dm, frn.spread)

    def test_discount_margin_discount(self):
        """Test DM for FRN trading at discount."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        # Price below par
        discount_price = frn.notional * 0.98  # 98%

        dm = self.engine.discount_margin(frn, discount_price)

        # DM should be greater than spread (trading cheap)
        self.assertGreater(dm, frn.spread)


class TestSimpleMargin(unittest.TestCase):
    """Test Simple Margin calculations."""

    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 1)
        self.rate_curve = FlatRateCurve(rate=0.05)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve, valuation_date=self.valuation_date
        )
        self.engine = FRNDiscountEngine(self.pricing_env)

    def test_simple_margin_at_par(self):
        """Test Simple Margin at par equals spread."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        sm = self.engine.simple_margin(frn, frn.notional)

        # At par, SM = spread
        self.assertAlmostEqual(sm, frn.spread, delta=0.001)

    def test_simple_margin_below_par(self):
        """Test Simple Margin for discount price."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        discount_price = frn.notional * 0.98

        sm = self.engine.simple_margin(frn, discount_price)

        # Below par adds to margin
        self.assertGreater(sm, frn.spread)


class TestFRNRiskMetrics(unittest.TestCase):
    """Test FRN risk metric calculations."""

    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 1)
        self.rate_curve = FlatRateCurve(rate=0.05)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve, valuation_date=self.valuation_date
        )
        self.engine = FRNDiscountEngine(self.pricing_env)

        self.frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

    def test_weighted_average_life(self):
        """Test WAL calculation."""
        wal = self.engine.weighted_average_life(self.frn)

        # For a 5-year bullet, WAL = 5 years
        self.assertAlmostEqual(wal, 5.0, delta=0.1)

    def test_effective_duration(self):
        """Test effective duration is low for FRN."""
        eff_dur = self.engine.effective_duration(self.frn)

        # FRN duration should be much lower than maturity
        # Typically close to time to next reset
        self.assertLess(eff_dur, 1.0)  # Less than 1 year

    def test_spread_duration(self):
        """Test spread duration is positive and reasonable."""
        spread_dur = self.engine.spread_duration(self.frn)
        wal = self.engine.weighted_average_life(self.frn)

        # Spread duration should be positive
        self.assertGreater(spread_dur, 0)

        # Spread duration should be in reasonable range relative to WAL
        # (could be different due to coupon effects)
        self.assertLess(spread_dur, wal + 1.0)

    def test_dv01(self):
        """Test DV01 calculation."""
        dv01 = self.engine.dv01(self.frn)

        # DV01 should be positive
        self.assertGreater(dv01, 0)

    def test_cs01(self):
        """Test CS01 calculation."""
        cs01 = self.engine.cs01(self.frn)

        # CS01 should be positive
        self.assertGreater(cs01, 0)


class TestYieldToMaturity(unittest.TestCase):
    """Test FRN Yield to Maturity calculations."""

    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 1)
        self.rate_curve = FlatRateCurve(rate=0.05)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve, valuation_date=self.valuation_date
        )
        self.engine = FRNDiscountEngine(self.pricing_env)

        self.frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,  # 50bp
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

    def test_ytm_at_par(self):
        """Test YTM at par price."""
        ytm = self.engine.yield_to_maturity(
            self.frn, self.frn.notional, clean_price=True
        )

        # YTM should be approximately index rate + spread
        expected_ytm = 0.05 + 0.005  # 5.5%
        self.assertAlmostEqual(ytm, expected_ytm, delta=0.005)

    def test_ytm_discount_vs_premium(self):
        """Test that YTM is higher for discount price, lower for premium."""
        par_price = self.frn.notional
        discount_price = self.frn.notional * 0.98
        premium_price = self.frn.notional * 1.02

        ytm_par = self.engine.yield_to_maturity(self.frn, par_price)
        ytm_discount = self.engine.yield_to_maturity(self.frn, discount_price)
        ytm_premium = self.engine.yield_to_maturity(self.frn, premium_price)

        # Discount price -> higher YTM
        self.assertGreater(ytm_discount, ytm_par)
        # Premium price -> lower YTM
        self.assertLess(ytm_premium, ytm_par)

    def test_ytm_with_assumed_rate(self):
        """Test YTM with explicit assumed index rate."""
        assumed_rate = 0.06  # 6%

        ytm = self.engine.yield_to_maturity(
            self.frn, self.frn.notional, assumed_index_rate=assumed_rate
        )

        # YTM should be approximately assumed rate + spread
        expected_ytm = assumed_rate + self.frn.spread
        self.assertAlmostEqual(ytm, expected_ytm, delta=0.005)

    def test_ytm_positive(self):
        """Test that YTM is positive for reasonable prices."""
        ytm = self.engine.yield_to_maturity(self.frn, self.frn.notional)
        self.assertGreater(ytm, 0)


class TestPriceFromYield(unittest.TestCase):
    """Test FRN price_from_yield calculations."""

    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 1)
        self.rate_curve = FlatRateCurve(rate=0.05)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve, valuation_date=self.valuation_date
        )
        self.engine = FRNDiscountEngine(self.pricing_env)

        self.frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

    def test_price_from_yield_at_index_plus_spread(self):
        """Test that price is near par when yield = index + spread."""
        ytm = 0.055  # 5% index + 0.5% spread
        price = self.engine.clean_price_from_yield(
            self.frn, ytm, assumed_index_rate=0.05
        )

        # Should be close to par
        self.assertAlmostEqual(price / self.frn.notional, 1.0, delta=0.02)

    def test_price_from_yield_inverse_relationship(self):
        """Test that higher yield gives lower price."""
        low_yield = 0.04
        high_yield = 0.07

        price_low = self.engine.clean_price_from_yield(
            self.frn, low_yield, assumed_index_rate=0.05
        )
        price_high = self.engine.clean_price_from_yield(
            self.frn, high_yield, assumed_index_rate=0.05
        )

        self.assertGreater(price_low, price_high)

    def test_dirty_vs_clean_price_from_yield(self):
        """Test that dirty price = clean price + accrued."""
        ytm = 0.055

        # Mid-period valuation
        mid_date = datetime(2024, 2, 15)
        pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve, valuation_date=mid_date
        )
        engine = FRNDiscountEngine(pricing_env)

        frn = create_simple_frn(
            issue_date=datetime(2023, 10, 1),
            maturity_date=datetime(2028, 10, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        # Add fixing for current period
        first_cf = frn.get_all_floating_cashflows()[0]
        frn.add_fixing(first_cf.fixing_date, 0.05)

        dirty = engine.dirty_price_from_yield(frn, ytm, mid_date, mid_date)
        clean = engine.clean_price_from_yield(frn, ytm, mid_date, mid_date)
        accrued = frn.calculate_accrued_interest(mid_date)

        self.assertAlmostEqual(dirty, clean + accrued, places=2)

    def test_price_from_yield_positive(self):
        """Test that prices are positive for reasonable yields."""
        for ytm in [0.03, 0.05, 0.07, 0.10]:
            price = self.engine.clean_price_from_yield(self.frn, ytm)
            self.assertGreater(price, 0)


class TestFullAnalysis(unittest.TestCase):
    """Test full FRN analysis."""

    def test_full_analysis_output(self):
        """Test that full analysis returns all metrics."""
        valuation_date = datetime(2024, 1, 1)
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05), valuation_date=valuation_date
        )
        engine = FRNDiscountEngine(pricing_env)

        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR_3M,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        results = engine.full_analysis(frn, market_price=frn.notional, clean_price=True)

        self.assertIsInstance(results, FRNPricingResults)
        self.assertIsNotNone(results.dirty_price)
        self.assertIsNotNone(results.clean_price)
        self.assertIsNotNone(results.accrued_interest)
        self.assertIsNotNone(results.discount_margin)
        self.assertIsNotNone(results.simple_margin)
        self.assertIsNotNone(results.yield_to_maturity)
        self.assertIsNotNone(results.effective_duration)
        self.assertIsNotNone(results.spread_duration)
        self.assertIsNotNone(results.weighted_average_life)
        self.assertIsNotNone(results.assumed_index_rate)


class TestDifferentIndices(unittest.TestCase):
    """Test FRNs with different reference indices."""

    def test_sofr_frn(self):
        """Test FRN with SOFR index."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,
            index=SOFR,
            spread=0.0050,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        self.assertEqual(frn.index.name, "SOFR")
        self.assertTrue(frn.index.is_overnight)

    def test_shibor_frn(self):
        """Test FRN with SHIBOR index (China market)."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=10000000.0,  # CNY
            index=SHIBOR_3M,
            spread=0.0080,  # 80bp
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        self.assertEqual(frn.index.name, "SHIBOR_3M")
        self.assertEqual(frn.index.currency, "CNY")

    def test_repo_7d_frn(self):
        """Test FRN with REPO 7D (DR007) index."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2027, 1, 1),
            notional=50000000.0,  # CNY
            index=REPO_7D,
            spread=0.0100,  # 100bp
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        self.assertEqual(frn.index.name, "REPO_7D")

    def test_euribor_frn(self):
        """Test FRN with EURIBOR index."""
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000000.0,  # EUR
            index=EURIBOR_3M,
            spread=0.0075,  # 75bp
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        self.assertEqual(frn.index.name, "EURIBOR_3M")
        self.assertEqual(frn.index.currency, "EUR")


if __name__ == "__main__":
    unittest.main()
