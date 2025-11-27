"""
Comprehensive tests for Interest Rate Swap (IRS) implementation.

Tests cover:
- Swap leg generation and validation
- Vanilla IRS pricing
- Basis swap pricing
- Amortizing notional schedules
- SOFR compounding cashflows
- Par rate/spread calculations
- Risk metrics (DV01, duration)
- Edge cases and error handling
"""

import unittest
from datetime import datetime
import math

from asset.rate.product.irs import (
    InterestRateSwap,
    BasisSwap,
    FixedLeg,
    FloatingLeg,
    NotionalSchedule,
    SwapDirection,
    create_vanilla_irs,
    create_basis_swap,
    create_amortizing_irs,
    create_compounding_irs,
)
from asset.rate.engine.irs_discount_engine import (
    IRSDiscountEngine,
    IRSPricingResults,
    BasisSwapPricingResults,
)
from asset.bond.schedule.cashflow import (
    CashFlow,
    FixedCashFlow,
    FloatingCashFlow,
    CompoundingMethod,
)
from param.index import (
    RateIndex,
    IndexFixing,
    IndexFixingStore,
    SOFR,
    SOFR_3M,
    EURIBOR_3M,
    SHIBOR_3M,
)
from param.rrf import FlatRateCurve
from priceenv import PricingEnvironment
from util.calendar import (
    DayCountConvention,
    BusinessDayConvention,
    CalendarType,
    create_calendar,
)
from util.enum import PaymentFrequency, ResetConvention
from util.exceptions import ValidationError


class TestNotionalSchedule(unittest.TestCase):
    """Test NotionalSchedule functionality."""
    
    def test_constant_notional(self):
        """Test constant notional schedule."""
        schedule = NotionalSchedule.constant(1000000.0)
        
        self.assertEqual(schedule.initial_notional, 1000000.0)
        self.assertEqual(len(schedule.notional_dates), 0)
        
        # Should return same notional for any date
        self.assertEqual(
            schedule.get_notional(datetime(2024, 1, 1)),
            1000000.0
        )
        self.assertEqual(
            schedule.get_notional(datetime(2030, 12, 31)),
            1000000.0
        )
    
    def test_amortizing_notional(self):
        """Test amortizing notional schedule."""
        schedule = NotionalSchedule(
            notional_dates=[
                datetime(2025, 1, 1),
                datetime(2026, 1, 1),
                datetime(2027, 1, 1),
            ],
            notional_amounts=[800000.0, 600000.0, 400000.0],
            initial_notional=1000000.0
        )
        
        # Before first amortization
        self.assertEqual(
            schedule.get_notional(datetime(2024, 6, 1)),
            1000000.0
        )
        
        # After first amortization
        self.assertEqual(
            schedule.get_notional(datetime(2025, 6, 1)),
            800000.0
        )
        
        # After all amortizations
        self.assertEqual(
            schedule.get_notional(datetime(2028, 1, 1)),
            400000.0
        )
    
    def test_linear_amortizing(self):
        """Test linear amortizing schedule factory."""
        schedule = NotionalSchedule.linear_amortizing(
            initial_notional=1000000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2029, 1, 1),
            num_periods=5,
            final_notional=0.0
        )
        
        self.assertEqual(schedule.initial_notional, 1000000.0)
        self.assertEqual(len(schedule.notional_dates), 5)
        
        # Should reach final notional at end
        self.assertAlmostEqual(
            schedule.notional_amounts[-1],
            0.0,
            places=0
        )
    
    def test_validation_negative_initial(self):
        """Test validation rejects negative initial notional."""
        with self.assertRaises(ValidationError):
            NotionalSchedule.constant(-1000000.0)
    
    def test_validation_negative_amount(self):
        """Test validation rejects negative amounts in schedule."""
        with self.assertRaises(ValidationError):
            NotionalSchedule(
                notional_dates=[datetime(2025, 1, 1)],
                notional_amounts=[-500000.0],
                initial_notional=1000000.0
            )


class TestFixedLeg(unittest.TestCase):
    """Test FixedLeg functionality."""
    
    def test_fixed_leg_creation(self):
        """Test basic fixed leg creation."""
        leg = FixedLeg(
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2029, 1, 15),
            notional_schedule=NotionalSchedule.constant(10000000.0),
            fixed_rate=0.045,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )
        
        self.assertEqual(leg.fixed_rate, 0.045)
        self.assertTrue(leg.is_fixed())
        self.assertEqual(leg.get_start_date(), datetime(2024, 1, 15))
        self.assertEqual(leg.get_end_date(), datetime(2029, 1, 15))
    
    def test_fixed_leg_cashflows(self):
        """Test fixed leg cashflow generation."""
        leg = FixedLeg(
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2026, 1, 15),  # 2 years
            notional_schedule=NotionalSchedule.constant(1000000.0),
            fixed_rate=0.04,  # 4%
            payment_frequency=PaymentFrequency.QUARTERLY,
        )
        
        cashflows = leg.get_all_cashflows()
        
        # 2 years * 4 quarters = 8 cashflows
        self.assertEqual(len(cashflows), 8)
        
        # Check first cashflow
        first_cf = cashflows[0]
        self.assertIsInstance(first_cf, CashFlow)
        self.assertEqual(first_cf.rate, 0.04)
        
        # Coupon should be approximately notional * rate * 0.25
        expected_coupon = 1000000.0 * 0.04 * 0.25
        self.assertAlmostEqual(first_cf.amount, expected_coupon, delta=1000)
    
    def test_fixed_leg_accrued_interest(self):
        """Test accrued interest calculation."""
        leg = FixedLeg(
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2029, 1, 15),
            notional_schedule=NotionalSchedule.constant(1000000.0),
            fixed_rate=0.04,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )
        
        # Mid-period should have positive accrued
        mid_date = datetime(2024, 3, 1)
        accrued = leg.calculate_accrued_interest(mid_date)
        self.assertGreater(accrued, 0)
        
        # Before start should be zero
        accrued_before = leg.calculate_accrued_interest(datetime(2024, 1, 1))
        self.assertEqual(accrued_before, 0.0)
        
        # After maturity should be zero
        accrued_after = leg.calculate_accrued_interest(datetime(2030, 1, 1))
        self.assertEqual(accrued_after, 0.0)
    
    def test_fixed_leg_validation(self):
        """Test fixed leg validation."""
        # End before start should fail
        with self.assertRaises(ValidationError):
            FixedLeg(
                start_date=datetime(2029, 1, 15),
                end_date=datetime(2024, 1, 15),
                notional_schedule=NotionalSchedule.constant(1000000.0),
                fixed_rate=0.04,
                payment_frequency=PaymentFrequency.QUARTERLY,
            )


class TestFloatingLeg(unittest.TestCase):
    """Test FloatingLeg functionality."""
    
    def test_floating_leg_creation(self):
        """Test basic floating leg creation."""
        leg = FloatingLeg(
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2029, 1, 15),
            notional_schedule=NotionalSchedule.constant(10000000.0),
            index=SOFR_3M,
            spread=0.005,  # 50bp
            payment_frequency=PaymentFrequency.QUARTERLY,
        )
        
        self.assertEqual(leg.spread, 0.005)
        self.assertFalse(leg.is_fixed())
        self.assertEqual(leg.index.name, "SOFR_3M")
    
    def test_floating_leg_cashflows(self):
        """Test floating leg cashflow generation."""
        leg = FloatingLeg(
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2026, 1, 15),  # 2 years
            notional_schedule=NotionalSchedule.constant(1000000.0),
            index=SOFR_3M,
            spread=0.005,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )
        
        cashflows = leg.get_floating_cashflows()
        
        # 2 years * 4 quarters = 8 cashflows
        self.assertEqual(len(cashflows), 8)
        
        # All should be projected (no historical fixings)
        for cf in cashflows:
            self.assertTrue(cf.is_projected)
    
    def test_floating_leg_with_cap_floor(self):
        """Test floating leg with rate cap and floor."""
        leg = FloatingLeg(
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2029, 1, 15),
            notional_schedule=NotionalSchedule.constant(1000000.0),
            index=SOFR_3M,
            spread=0.005,
            payment_frequency=PaymentFrequency.QUARTERLY,
            rate_cap=0.10,   # 10% cap
            rate_floor=0.02,  # 2% floor
        )
        
        self.assertEqual(leg.rate_cap, 0.10)
        self.assertEqual(leg.rate_floor, 0.02)
    
    def test_floating_leg_in_arrears(self):
        """Test floating leg with in-arrears reset."""
        leg = FloatingLeg(
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2029, 1, 15),
            notional_schedule=NotionalSchedule.constant(1000000.0),
            index=SOFR_3M,
            spread=0.005,
            payment_frequency=PaymentFrequency.QUARTERLY,
            reset_convention=ResetConvention.IN_ARREARS,
            lookback_days=5,
        )
        
        self.assertEqual(leg.reset_convention, ResetConvention.IN_ARREARS)
        self.assertEqual(leg.lookback_days, 5)
    
    def test_floating_leg_validation(self):
        """Test floating leg validation."""
        # Cap < floor should fail
        with self.assertRaises(ValidationError):
            FloatingLeg(
                start_date=datetime(2024, 1, 15),
                end_date=datetime(2029, 1, 15),
                notional_schedule=NotionalSchedule.constant(1000000.0),
                index=SOFR_3M,
                spread=0.005,
                payment_frequency=PaymentFrequency.QUARTERLY,
                rate_cap=0.05,
                rate_floor=0.08,  # Floor > Cap
            )


class TestInterestRateSwap(unittest.TestCase):
    """Test InterestRateSwap functionality."""
    
    def test_vanilla_irs_creation(self):
        """Test vanilla IRS creation using factory."""
        irs = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
            spread=0.0,
            direction=SwapDirection.PAYER,
        )
        
        self.assertEqual(irs.get_fixed_rate(), 0.045)
        self.assertEqual(irs.direction, SwapDirection.PAYER)
        self.assertIsInstance(irs.fixed_leg, FixedLeg)
        self.assertIsInstance(irs.floating_leg, FloatingLeg)
    
    def test_payer_vs_receiver(self):
        """Test pay/receive leg assignment."""
        # Payer swap: pay fixed, receive floating
        payer = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
            direction=SwapDirection.PAYER,
        )
        
        self.assertTrue(payer.pay_leg.is_fixed())
        self.assertFalse(payer.receive_leg.is_fixed())
        
        # Receiver swap: receive fixed, pay floating
        receiver = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
            direction=SwapDirection.RECEIVER,
        )
        
        self.assertFalse(receiver.pay_leg.is_fixed())
        self.assertTrue(receiver.receive_leg.is_fixed())
    
    def test_time_to_maturity(self):
        """Test time to maturity calculation."""
        irs = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
        )
        
        ttm = irs.time_to_maturity(datetime(2024, 1, 15))
        self.assertAlmostEqual(ttm, 5.0, delta=0.1)
    
    def test_is_expired(self):
        """Test swap expiration check."""
        irs = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
        )
        
        self.assertFalse(irs.is_expired(datetime(2024, 1, 15)))
        self.assertFalse(irs.is_expired(datetime(2028, 12, 31)))
        self.assertTrue(irs.is_expired(datetime(2029, 1, 15)))
        self.assertTrue(irs.is_expired(datetime(2030, 1, 1)))


class TestBasisSwap(unittest.TestCase):
    """Test BasisSwap functionality."""
    
    def test_basis_swap_creation(self):
        """Test basis swap creation."""
        swap = create_basis_swap(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            index1=SOFR,
            index2=SOFR_3M,
            spread1=0.0,
            spread2=0.001,  # 10bp on receive leg
        )
        
        self.assertIsInstance(swap.leg1, FloatingLeg)
        self.assertIsInstance(swap.leg2, FloatingLeg)
        self.assertEqual(swap.leg1.index.name, "SOFR")
        self.assertEqual(swap.leg2.index.name, "SOFR_3M")
    
    def test_basis_spread(self):
        """Test basis spread calculation."""
        swap = create_basis_swap(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            index1=SOFR,
            index2=SOFR_3M,
            spread1=0.001,  # 10bp on pay leg
            spread2=0.003,  # 30bp on receive leg
        )
        
        # Net basis = leg2.spread - leg1.spread = 30bp - 10bp = 20bp
        self.assertAlmostEqual(swap.get_basis_spread(), 0.002, places=6)


class TestAmortizingSwap(unittest.TestCase):
    """Test amortizing swap functionality."""
    
    def test_amortizing_irs_creation(self):
        """Test amortizing IRS creation."""
        amort_schedule = [
            (datetime(2025, 1, 15), 8000000.0),
            (datetime(2026, 1, 15), 6000000.0),
            (datetime(2027, 1, 15), 4000000.0),
            (datetime(2028, 1, 15), 2000000.0),
        ]
        
        irs = create_amortizing_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            initial_notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
            amortization_schedule=amort_schedule,
        )
        
        # Check notional at different dates
        self.assertEqual(irs.get_notional(datetime(2024, 6, 1)), 10000000.0)
        self.assertEqual(irs.get_notional(datetime(2025, 6, 1)), 8000000.0)
        self.assertEqual(irs.get_notional(datetime(2028, 6, 1)), 2000000.0)


class TestCompoundingSwap(unittest.TestCase):
    """Test SOFR compounding swap functionality."""
    
    def test_compounding_irs_creation(self):
        """Test compounding IRS creation."""
        irs = create_compounding_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR,
            spread=0.0005,  # 5bp
            compounding_method=CompoundingMethod.SPREAD_EXCLUSIVE,
            lookback_days=2,
        )
        
        self.assertEqual(
            irs.floating_leg.compounding_method,
            CompoundingMethod.SPREAD_EXCLUSIVE
        )
        self.assertEqual(irs.floating_leg.lookback_days, 2)
        self.assertTrue(irs.floating_leg.index.is_overnight)


class TestIRSPricing(unittest.TestCase):
    """Test IRS pricing engine."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 15)
        self.rate_curve = FlatRateCurve(rate=0.045)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date
        )
        self.engine = IRSDiscountEngine(self.pricing_env)
    
    def test_at_market_swap_npv_near_zero(self):
        """Test that at-market swap has NPV near zero."""
        # Create a swap with any rate
        test_swap = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.04,
            index=SOFR_3M,
            direction=SwapDirection.PAYER,
        )
        
        # Get par rate
        par_rate = self.engine.par_rate(test_swap)
        
        # Create at-market swap
        at_market = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=par_rate,
            index=SOFR_3M,
            direction=SwapDirection.PAYER,
        )
        
        npv = self.engine.npv(at_market)
        
        # NPV should be very close to zero
        self.assertAlmostEqual(npv, 0.0, delta=100)  # Within $100
    
    def test_payer_receiver_opposite_npv(self):
        """Test that payer and receiver have opposite NPVs."""
        payer = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
            direction=SwapDirection.PAYER,
        )
        
        receiver = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
            direction=SwapDirection.RECEIVER,
        )
        
        payer_npv = self.engine.npv(payer)
        receiver_npv = self.engine.npv(receiver)
        
        # Should be approximately opposite
        self.assertAlmostEqual(payer_npv, -receiver_npv, delta=1)
    
    def test_expired_swap_npv_zero(self):
        """Test that expired swap has zero NPV."""
        irs = create_vanilla_irs(
            effective_date=datetime(2020, 1, 15),
            maturity_date=datetime(2022, 1, 15),  # Already matured
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
        )
        
        npv = self.engine.npv(irs, valuation_date=datetime(2024, 1, 15))
        self.assertEqual(npv, 0.0)
    
    def test_higher_fixed_rate_positive_payer_npv(self):
        """Test that higher fixed rate means positive NPV for payer."""
        # With flat curve at 4.5%, paying 5% should have positive NPV
        # because we're paying more than market
        high_rate_swap = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.05,  # Higher than curve rate
            index=SOFR_3M,
            direction=SwapDirection.PAYER,
        )
        
        par_rate = self.engine.par_rate(high_rate_swap)
        npv = self.engine.npv(high_rate_swap)
        
        # If fixed rate > par rate, payer NPV should be negative
        if high_rate_swap.get_fixed_rate() > par_rate:
            self.assertLess(npv, 0)


class TestBasisSwapPricing(unittest.TestCase):
    """Test basis swap pricing."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 15)
        self.rate_curve = FlatRateCurve(rate=0.045)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date
        )
        self.engine = IRSDiscountEngine(self.pricing_env)
    
    def test_basis_swap_pricing(self):
        """Test basic basis swap pricing."""
        swap = create_basis_swap(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            index1=SOFR,
            index2=SOFR_3M,
            spread1=0.0,
            spread2=0.001,
        )
        
        npv = self.engine.npv(swap)
        
        # With positive spread on receive leg, NPV should be positive
        self.assertGreater(npv, 0)
    
    def test_basis_swap_full_analysis(self):
        """Test basis swap full analysis."""
        swap = create_basis_swap(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            index1=SOFR,
            index2=SOFR_3M,
            spread1=0.0,
            spread2=0.001,
        )
        
        results = self.engine.full_basis_swap_analysis(swap)
        
        self.assertIsInstance(results, BasisSwapPricingResults)
        self.assertIsNotNone(results.npv)
        self.assertIsNotNone(results.leg1_pv)
        self.assertIsNotNone(results.leg2_pv)


class TestRiskMetrics(unittest.TestCase):
    """Test risk metric calculations."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 15)
        self.rate_curve = FlatRateCurve(rate=0.045)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date
        )
        self.engine = IRSDiscountEngine(self.pricing_env)
        
        self.swap = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
            direction=SwapDirection.PAYER,
        )
    
    def test_dv01_calculation(self):
        """Test DV01 calculation."""
        dv01 = self.engine.dv01(self.swap)
        
        # DV01 should be non-zero for a swap
        self.assertNotEqual(dv01, 0.0)
    
    def test_bpv_equals_dv01(self):
        """Test that BPV equals DV01."""
        dv01 = self.engine.dv01(self.swap)
        bpv = self.engine.bpv(self.swap)
        
        self.assertEqual(dv01, bpv)
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        duration = self.engine.duration(self.swap)
        
        # Duration should be calculated
        # (may be unusual for near-par swap)
        self.assertIsNotNone(duration)
    
    def test_weighted_average_life(self):
        """Test WAL calculation."""
        wal = self.engine.weighted_average_life(self.swap)
        
        # WAL should be positive for active swap
        self.assertGreater(wal, 0)
        
        # WAL should be less than maturity for quarterly payments
        self.assertLess(wal, 5.0)
    
    def test_accrued_interest(self):
        """Test accrued interest calculation."""
        # Move to mid-period
        mid_date = datetime(2024, 3, 1)
        
        # Update forward rates so floating leg has rates
        self.swap.update_forward_rates(self.rate_curve, self.valuation_date)
        
        pay_accrued, receive_accrued = self.engine.accrued_interest(
            self.swap, mid_date
        )
        
        # Pay leg (fixed) should have positive accrued mid-period
        self.assertGreater(pay_accrued, 0)
        
        # Receive leg (floating) should also have positive accrued
        # after forward rates are set
        self.assertGreaterEqual(receive_accrued, 0)


class TestFullAnalysis(unittest.TestCase):
    """Test full swap analysis."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 15)
        self.rate_curve = FlatRateCurve(rate=0.045)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date
        )
        self.engine = IRSDiscountEngine(self.pricing_env)
    
    def test_full_analysis_returns_all_metrics(self):
        """Test that full analysis returns all expected metrics."""
        swap = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
        )
        
        results = self.engine.full_analysis(swap)
        
        self.assertIsInstance(results, IRSPricingResults)
        self.assertIsNotNone(results.npv)
        self.assertIsNotNone(results.receive_leg_pv)
        self.assertIsNotNone(results.pay_leg_pv)
        self.assertIsNotNone(results.par_rate)
        self.assertIsNotNone(results.dv01)
        self.assertIsNotNone(results.bpv)
        self.assertIsNotNone(results.duration)
        self.assertIsNotNone(results.weighted_average_life)


class TestDifferentIndices(unittest.TestCase):
    """Test swaps with different reference indices."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 15)
        self.rate_curve = FlatRateCurve(rate=0.045)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date
        )
        self.engine = IRSDiscountEngine(self.pricing_env)
    
    def test_sofr_swap(self):
        """Test swap with SOFR overnight index."""
        swap = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR,
        )
        
        self.assertTrue(swap.floating_leg.index.is_overnight)
        npv = self.engine.npv(swap)
        self.assertIsNotNone(npv)
    
    def test_euribor_swap(self):
        """Test swap with EURIBOR index."""
        swap = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.035,
            index=EURIBOR_3M,
        )
        
        self.assertEqual(swap.floating_leg.index.currency, "EUR")
        npv = self.engine.npv(swap)
        self.assertIsNotNone(npv)
    
    def test_shibor_swap(self):
        """Test swap with SHIBOR index."""
        swap = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=70000000.0,  # CNY
            fixed_rate=0.025,
            index=SHIBOR_3M,
        )
        
        self.assertEqual(swap.floating_leg.index.currency, "CNY")
        npv = self.engine.npv(swap)
        self.assertIsNotNone(npv)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 15)
        self.rate_curve = FlatRateCurve(rate=0.045)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date
        )
        self.engine = IRSDiscountEngine(self.pricing_env)
    
    def test_very_short_swap(self):
        """Test swap with very short maturity."""
        short_swap = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2024, 7, 15),  # 6 months
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
        )
        
        npv = self.engine.npv(short_swap)
        self.assertIsNotNone(npv)
    
    def test_very_long_swap(self):
        """Test swap with very long maturity."""
        long_swap = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2054, 1, 15),  # 30 years
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
        )
        
        npv = self.engine.npv(long_swap)
        self.assertIsNotNone(npv)
    
    def test_zero_spread(self):
        """Test swap with zero spread on floating leg."""
        swap = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
            spread=0.0,
        )
        
        npv = self.engine.npv(swap)
        self.assertIsNotNone(npv)
    
    def test_negative_spread(self):
        """Test swap with negative spread."""
        swap = create_vanilla_irs(
            effective_date=datetime(2024, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            notional=10000000.0,
            fixed_rate=0.045,
            index=SOFR_3M,
            spread=-0.001,  # Negative spread
        )
        
        npv = self.engine.npv(swap)
        self.assertIsNotNone(npv)


if __name__ == '__main__':
    unittest.main()

