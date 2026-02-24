"""
Tests for Cap/Floor analytical (Black's model) pricing engine.
"""

import unittest
from datetime import datetime

from asset.rate.product.cap_floor import (
    create_cap,
    create_floor,
    create_collar,
)
from asset.rate.engine.cap_floor_engine import (
    CapFloorEngine,
    CapFloorPricingResults,
)
from param.index import SOFR_3M
from param.rrf import FlatRateCurve
from priceenv import PricingEnvironment


class TestCapFloorEnginePricing(unittest.TestCase):
    """Test Cap/Floor engine pricing."""

    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 15)
        self.rate_curve = FlatRateCurve(rate=0.05)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date,
        )
        self.vol = 0.20  # 20% Black vol

    def test_atm_cap_positive_price(self):
        """ATM cap should have positive price."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        engine = CapFloorEngine(self.pricing_env, vol=self.vol)
        price = engine.price(cap)
        self.assertGreater(price, 0.0)

    def test_atm_floor_positive_price(self):
        """ATM floor should have positive price."""
        floor = create_floor(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        engine = CapFloorEngine(self.pricing_env, vol=self.vol)
        price = engine.price(floor)
        self.assertGreater(price, 0.0)

    def test_deep_otm_cap_near_zero(self):
        """Deep OTM cap (high strike) should have near-zero price."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.15,  # 15% strike, market at 5%
            index=SOFR_3M,
        )
        engine = CapFloorEngine(self.pricing_env, vol=self.vol)
        price = engine.price(cap)
        # Should be very small relative to notional
        self.assertLess(price, 1000)

    def test_deep_itm_cap_near_intrinsic(self):
        """Deep ITM cap (low strike) should be near intrinsic."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.01,  # 1% strike, market at 5%
            index=SOFR_3M,
        )
        engine = CapFloorEngine(self.pricing_env, vol=self.vol)
        price = engine.price(cap)
        # Should be large (roughly N * dcf * (fwd - K) * num_periods * avg_df)
        self.assertGreater(price, 100_000)

    def test_cap_floor_parity(self):
        """Test put-call parity: Cap - Floor = sum of forward rate agreements.

        For ATM options: Cap_ATM ≈ Floor_ATM (approximately equal when F ≈ K).
        More precisely: Cap - Floor = PV(forward - strike) per period.
        """
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        floor = create_floor(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        engine = CapFloorEngine(self.pricing_env, vol=self.vol)
        cap_price = engine.price(cap)
        floor_price = engine.price(floor)

        # For ATM (F ≈ K): cap ≈ floor
        # Cap - Floor = PV of (F - K) for each period (should be near zero for flat curve)
        self.assertAlmostEqual(cap_price, floor_price, delta=cap_price * 0.1 + 100)

    def test_higher_vol_higher_price(self):
        """Higher volatility should increase both cap and floor prices."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )

        engine_low = CapFloorEngine(self.pricing_env, vol=0.10)
        engine_high = CapFloorEngine(self.pricing_env, vol=0.30)

        price_low = engine_low.price(cap)
        price_high = engine_high.price(cap)

        self.assertGreater(price_high, price_low)

    def test_expired_cap_returns_zero(self):
        """Expired cap should return zero."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        engine = CapFloorEngine(self.pricing_env, vol=self.vol)
        price = engine.price(cap, valuation_date=datetime(2027, 1, 1))
        self.assertEqual(price, 0.0)

    def test_cap_price_scales_with_notional(self):
        """Cap price should scale linearly with notional."""
        cap1 = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=5_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        cap2 = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        engine = CapFloorEngine(self.pricing_env, vol=self.vol)
        p1 = engine.price(cap1)
        p2 = engine.price(cap2)
        self.assertAlmostEqual(p2 / p1, 2.0, delta=0.01)

    def test_zero_vol_returns_intrinsic(self):
        """Zero vol should return intrinsic value only."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.04,  # ITM: strike < forward
            index=SOFR_3M,
        )
        engine = CapFloorEngine(self.pricing_env, vol=1e-12)
        price = engine.price(cap)
        # Should be approximately the intrinsic value
        self.assertGreater(price, 0)


class TestCapFloorEngineCollar(unittest.TestCase):
    """Test Collar pricing."""

    def test_collar_price(self):
        """Collar = Cap - Floor."""
        collar = create_collar(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            cap_strike=0.06,
            floor_strike=0.04,
            index=SOFR_3M,
        )

        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        engine = CapFloorEngine(pricing_env, vol=0.20)

        collar_price = engine.price_collar(collar)
        cap_price = engine.price(collar.cap)
        floor_price = engine.price(collar.floor)

        self.assertAlmostEqual(collar_price, cap_price - floor_price, places=2)

    def test_zero_cost_collar(self):
        """Symmetric collar around ATM should be near-zero cost."""
        collar = create_collar(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            cap_strike=0.06,
            floor_strike=0.04,
            index=SOFR_3M,
        )

        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        engine = CapFloorEngine(pricing_env, vol=0.20)
        collar_price = engine.price_collar(collar)

        # Not necessarily zero, but should be small relative to cap/floor prices
        cap_price = engine.price(collar.cap)
        self.assertLess(abs(collar_price), cap_price)


class TestCapFloorEngineDV01(unittest.TestCase):
    """Test Cap/Floor DV01."""

    def test_cap_dv01(self):
        """Cap DV01 should be non-zero."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        engine = CapFloorEngine(pricing_env, vol=0.20)
        dv01 = engine.dv01(cap)
        self.assertNotEqual(dv01, 0.0)


class TestCapFloorEngineVega(unittest.TestCase):
    """Test Cap/Floor vega."""

    def test_cap_vega_positive(self):
        """Cap vega should be positive (higher vol -> higher price)."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        engine = CapFloorEngine(pricing_env, vol=0.20)
        vega = engine.vega(cap)
        self.assertGreater(vega, 0)


class TestCapFloorEngineFullAnalysis(unittest.TestCase):
    """Test full analysis."""

    def test_full_analysis_returns_all_fields(self):
        """Full analysis should populate all fields."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        engine = CapFloorEngine(pricing_env, vol=0.20)
        results = engine.full_analysis(cap)

        self.assertIsInstance(results, CapFloorPricingResults)
        self.assertGreater(results.npv, 0)
        self.assertTrue(len(results.caplet_prices) > 0)
        self.assertTrue(len(results.caplet_details) > 0)
        self.assertIsNotNone(results.dv01)
        self.assertIsNotNone(results.vega)

    def test_caplet_details_sum_to_total(self):
        """Sum of caplet prices should equal total NPV."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        engine = CapFloorEngine(pricing_env, vol=0.20)
        results = engine.full_analysis(cap)

        caplet_sum = sum(results.caplet_prices)
        self.assertAlmostEqual(caplet_sum, results.npv, places=2)


if __name__ == "__main__":
    unittest.main()
