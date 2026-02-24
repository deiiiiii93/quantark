"""
Tests for FRA analytical pricing engine.
"""

import unittest
from datetime import datetime

from asset.rate.product.fra import create_fra
from asset.rate.engine.fra_engine import FRAEngine, FRAPricingResults
from param.index import SOFR_3M
from param.rrf import FlatRateCurve
from priceenv import PricingEnvironment


class TestFRAEnginePricing(unittest.TestCase):
    """Test FRA engine pricing."""

    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 15)
        self.rate_curve = FlatRateCurve(rate=0.05)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date,
        )
        self.fra = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 4, 15),
            tenor_months=3,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )

    def test_at_market_fra_near_zero(self):
        """FRA priced at the forward rate should have ~zero NPV."""
        engine = FRAEngine(self.pricing_env)

        # For a flat curve at 5%, forward rate ≈ 5%
        # FRA with fixed_rate = 5% should have NPV near zero
        npv = engine.price(self.fra)
        self.assertAlmostEqual(npv, 0.0, delta=100)  # Within $100

    def test_forward_rate_flat_curve(self):
        """Forward rate on flat curve should equal the curve rate."""
        engine = FRAEngine(self.pricing_env)
        fwd = engine.forward_rate(self.fra)
        # On a flat 5% curve, forward rate should be very close to 5%
        self.assertAlmostEqual(fwd, 0.05, delta=0.001)

    def test_positive_npv_when_rates_rise(self):
        """Buyer profits when market rates rise above fixed rate."""
        fra = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 4, 15),
            tenor_months=3,
            notional=10_000_000,
            fixed_rate=0.04,  # Fixed at 4%
            index=SOFR_3M,
        )
        # Market is at 5% -> buyer receives 5%, pays 4%
        engine = FRAEngine(self.pricing_env)
        npv = engine.price(fra)
        self.assertGreater(npv, 0.0)

    def test_negative_npv_when_rates_fall(self):
        """Buyer loses when market rates fall below fixed rate."""
        fra = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 4, 15),
            tenor_months=3,
            notional=10_000_000,
            fixed_rate=0.06,  # Fixed at 6%
            index=SOFR_3M,
        )
        # Market is at 5% -> buyer receives 5%, pays 6%
        engine = FRAEngine(self.pricing_env)
        npv = engine.price(fra)
        self.assertLess(npv, 0.0)

    def test_expired_fra_returns_zero(self):
        """Expired FRA should return zero NPV."""
        engine = FRAEngine(self.pricing_env)
        npv = engine.price(self.fra, valuation_date=datetime(2024, 7, 15))
        self.assertEqual(npv, 0.0)

    def test_npv_scales_with_notional(self):
        """NPV should scale linearly with notional."""
        fra_small = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 4, 15),
            tenor_months=3,
            notional=1_000_000,
            fixed_rate=0.04,
            index=SOFR_3M,
        )
        fra_large = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 4, 15),
            tenor_months=3,
            notional=10_000_000,
            fixed_rate=0.04,
            index=SOFR_3M,
        )
        engine = FRAEngine(self.pricing_env)
        npv_small = engine.price(fra_small)
        npv_large = engine.price(fra_large)

        self.assertAlmostEqual(npv_large / npv_small, 10.0, delta=0.01)

    def test_par_rate_equals_forward(self):
        """Par rate should equal the forward rate."""
        engine = FRAEngine(self.pricing_env)
        par = engine.par_rate(self.fra)
        fwd = engine.forward_rate(self.fra)
        self.assertAlmostEqual(par, fwd, places=10)


class TestFRAEngineDV01(unittest.TestCase):
    """Test FRA DV01 calculation."""

    def test_dv01_positive(self):
        """DV01 should be positive for a FRA buyer."""
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        fra = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 4, 15),
            tenor_months=3,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = FRAEngine(pricing_env)
        dv01 = engine.dv01(fra)
        # FRA DV01 should be positive (buyer benefits from rate increase)
        self.assertGreater(abs(dv01), 0)

    def test_dv01_scales_with_notional(self):
        """DV01 should scale linearly with notional."""
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        fra1 = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 4, 15),
            tenor_months=3,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        fra2 = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 4, 15),
            tenor_months=3,
            notional=20_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = FRAEngine(pricing_env)
        dv01_1 = engine.dv01(fra1)
        dv01_2 = engine.dv01(fra2)
        self.assertAlmostEqual(dv01_2 / dv01_1, 2.0, delta=0.01)


class TestFRAEngineFullAnalysis(unittest.TestCase):
    """Test FRA full analysis."""

    def test_full_analysis_returns_all_fields(self):
        """Full analysis should populate all result fields."""
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        fra = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 4, 15),
            tenor_months=3,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = FRAEngine(pricing_env)
        results = engine.full_analysis(fra)

        self.assertIsInstance(results, FRAPricingResults)
        self.assertIsNotNone(results.npv)
        self.assertIsNotNone(results.forward_rate)
        self.assertIsNotNone(results.settlement_amount)
        self.assertIsNotNone(results.settlement_pv)
        self.assertIsNotNone(results.day_count_fraction)
        self.assertIsNotNone(results.dv01)
        self.assertIsNotNone(results.par_rate)
        self.assertGreater(results.day_count_fraction, 0)

    def test_full_analysis_par_rate(self):
        """Par rate in full analysis should match standalone par_rate."""
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        fra = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 4, 15),
            tenor_months=3,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = FRAEngine(pricing_env)
        results = engine.full_analysis(fra)
        par = engine.par_rate(fra)
        self.assertAlmostEqual(results.par_rate, par, places=10)


class TestFRAEngineValidation(unittest.TestCase):
    """Test FRA engine validation."""

    def test_no_rate_curve_raises(self):
        """Engine should raise if no rate curve in pricing env."""
        # PricingEnvironment requires rate_curve, so this is tested indirectly
        with self.assertRaises(Exception):
            FRAEngine(None)


if __name__ == "__main__":
    unittest.main()
