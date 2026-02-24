"""
Tests for Swaption analytical (Black's model) pricing engine.
"""

import unittest
from datetime import datetime

from asset.rate.product.swaption import (
    create_payer_swaption,
    create_receiver_swaption,
)
from asset.rate.engine.swaption_engine import (
    SwaptionEngine,
    SwaptionPricingResults,
    SwaptionModelType,
)
from param.index import SOFR_3M
from param.rrf import FlatRateCurve
from priceenv import PricingEnvironment


class TestSwaptionEnginePricing(unittest.TestCase):
    """Test Swaption engine pricing."""

    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 15)
        self.rate_curve = FlatRateCurve(rate=0.05)
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date,
        )
        self.vol = 0.20  # 20% Black vol

    def test_atm_payer_swaption_positive(self):
        """ATM payer swaption should have positive price."""
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,  # ATM (flat curve at 5%)
            index=SOFR_3M,
        )
        engine = SwaptionEngine(self.pricing_env, vol=self.vol)
        price = engine.price(swaption)
        self.assertGreater(price, 0.0)

    def test_atm_receiver_swaption_positive(self):
        """ATM receiver swaption should have positive price."""
        swaption = create_receiver_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = SwaptionEngine(self.pricing_env, vol=self.vol)
        price = engine.price(swaption)
        self.assertGreater(price, 0.0)

    def test_payer_receiver_parity(self):
        """Test put-call parity: Payer - Receiver = PV(forward swap).

        At ATM on flat curve: Payer ≈ Receiver.
        More precisely: Payer - Receiver = Annuity * (S - K)
        """
        swaption_pay = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        swaption_rec = create_receiver_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )

        engine = SwaptionEngine(self.pricing_env, vol=self.vol)
        payer_price = engine.price(swaption_pay)
        receiver_price = engine.price(swaption_rec)

        # For ATM (S ≈ K): payer ≈ receiver
        # The difference = A * (S - K) which should be near zero on flat curve
        diff = payer_price - receiver_price
        avg = (payer_price + receiver_price) / 2
        self.assertAlmostEqual(diff / avg, 0.0, delta=0.1)

    def test_deep_otm_payer_near_zero(self):
        """Deep OTM payer swaption (high strike) should be near-zero."""
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.15,  # 15% strike, market at 5%
            index=SOFR_3M,
        )
        engine = SwaptionEngine(self.pricing_env, vol=self.vol)
        price = engine.price(swaption)
        self.assertLess(price, 1000)

    def test_deep_itm_payer_near_intrinsic(self):
        """Deep ITM payer swaption (low strike) should be near intrinsic."""
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.01,  # 1% strike, market at 5%
            index=SOFR_3M,
        )
        engine = SwaptionEngine(self.pricing_env, vol=self.vol)
        price = engine.price(swaption)
        # Deep ITM: should be at least Annuity * (S - K)
        self.assertGreater(price, 100_000)

    def test_higher_vol_higher_price(self):
        """Higher vol should increase swaption price."""
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )

        engine_low = SwaptionEngine(self.pricing_env, vol=0.10)
        engine_high = SwaptionEngine(self.pricing_env, vol=0.30)

        price_low = engine_low.price(swaption)
        price_high = engine_high.price(swaption)

        self.assertGreater(price_high, price_low)

    def test_longer_option_tenor_higher_price(self):
        """Longer option tenor should increase swaption price (ATM)."""
        swaption_short = create_payer_swaption(
            exercise_date=datetime(2024, 7, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        swaption_long = create_payer_swaption(
            exercise_date=datetime(2026, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )

        engine = SwaptionEngine(self.pricing_env, vol=self.vol)
        price_short = engine.price(swaption_short)
        price_long = engine.price(swaption_long)

        self.assertGreater(price_long, price_short)

    def test_expired_swaption_returns_zero(self):
        """Expired swaption should return zero."""
        swaption = create_payer_swaption(
            exercise_date=datetime(2024, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
            trade_date=datetime(2023, 1, 1),
        )
        engine = SwaptionEngine(self.pricing_env, vol=self.vol)
        price = engine.price(swaption, valuation_date=datetime(2025, 1, 1))
        self.assertEqual(price, 0.0)

    def test_price_scales_with_notional(self):
        """Price should scale linearly with notional."""
        swaption1 = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=5_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        swaption2 = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = SwaptionEngine(self.pricing_env, vol=self.vol)
        p1 = engine.price(swaption1)
        p2 = engine.price(swaption2)
        self.assertAlmostEqual(p2 / p1, 2.0, delta=0.01)


class TestSwaptionEngineBachelier(unittest.TestCase):
    """Test Bachelier (normal) model."""

    def setUp(self):
        self.pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )

    def test_bachelier_positive_price(self):
        """Bachelier model should produce positive ATM price."""
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        # Normal vol: typically 50-100bp for rates
        engine = SwaptionEngine(
            self.pricing_env,
            vol=0.008,  # 80bp normal vol
            model=SwaptionModelType.BACHELIER,
        )
        price = engine.price(swaption)
        self.assertGreater(price, 0.0)

    def test_bachelier_payer_receiver_parity(self):
        """Bachelier payer-receiver parity: Payer - Receiver = A*(S-K)."""
        swaption_pay = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        swaption_rec = create_receiver_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = SwaptionEngine(
            self.pricing_env,
            vol=0.008,
            model=SwaptionModelType.BACHELIER,
        )
        diff = engine.price(swaption_pay) - engine.price(swaption_rec)
        avg = (engine.price(swaption_pay) + engine.price(swaption_rec)) / 2
        # ATM: S ≈ K, so diff ≈ 0 (small due to day-count discretization)
        self.assertAlmostEqual(diff / avg, 0.0, delta=0.15)


class TestSwaptionEngineForwardRate(unittest.TestCase):
    """Test forward swap rate and annuity calculation."""

    def test_forward_rate_flat_curve(self):
        """Forward swap rate on flat curve should equal curve rate."""
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = SwaptionEngine(pricing_env, vol=0.20)
        fwd = engine.forward_swap_rate(swaption)
        self.assertAlmostEqual(fwd, 0.05, delta=0.002)

    def test_annuity_positive(self):
        """Annuity should be positive."""
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = SwaptionEngine(pricing_env, vol=0.20)
        annuity = engine.annuity(swaption)
        self.assertGreater(annuity, 0.0)

    def test_annuity_scales_with_notional(self):
        """Annuity should scale with notional."""
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        swaption1 = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        swaption2 = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=20_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = SwaptionEngine(pricing_env, vol=0.20)
        a1 = engine.annuity(swaption1)
        a2 = engine.annuity(swaption2)
        self.assertAlmostEqual(a2 / a1, 2.0, delta=0.01)


class TestSwaptionEngineDV01(unittest.TestCase):
    """Test DV01."""

    def test_dv01_nonzero(self):
        """DV01 should be non-zero for a swaption."""
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = SwaptionEngine(pricing_env, vol=0.20)
        dv01 = engine.dv01(swaption)
        self.assertNotEqual(dv01, 0.0)


class TestSwaptionEngineVega(unittest.TestCase):
    """Test vega."""

    def test_vega_positive(self):
        """Swaption vega should be positive."""
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = SwaptionEngine(pricing_env, vol=0.20)
        vega = engine.vega(swaption)
        self.assertGreater(vega, 0)


class TestSwaptionEngineFullAnalysis(unittest.TestCase):
    """Test full analysis."""

    def test_full_analysis_returns_all_fields(self):
        """Full analysis should populate all fields."""
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = SwaptionEngine(pricing_env, vol=0.20)
        results = engine.full_analysis(swaption)

        self.assertIsInstance(results, SwaptionPricingResults)
        self.assertGreater(results.npv, 0)
        self.assertGreater(results.forward_swap_rate, 0)
        self.assertGreater(results.annuity, 0)
        self.assertIsNotNone(results.implied_vol)
        self.assertIsNotNone(results.intrinsic)
        self.assertIsNotNone(results.time_value)
        self.assertIsNotNone(results.delta)
        self.assertIsNotNone(results.vega)
        self.assertIsNotNone(results.dv01)

    def test_time_value_non_negative(self):
        """Time value should be non-negative."""
        pricing_env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.05),
            valuation_date=datetime(2024, 1, 15),
        )
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 1, 15),
            swap_tenor_years=5,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )
        engine = SwaptionEngine(pricing_env, vol=0.20)
        results = engine.full_analysis(swaption)
        self.assertGreaterEqual(results.time_value, -1)  # Allow tiny numerical noise


if __name__ == "__main__":
    unittest.main()
