"""
Tests for Swaption product.
"""

import unittest
from datetime import datetime

from asset.rate.product.swaption import (
    Swaption,
    SwaptionType,
    SwaptionExerciseStyle,
    create_payer_swaption,
    create_receiver_swaption,
)
from asset.rate.product.irs import InterestRateSwap, SwapDirection
from param.index import SOFR_3M, EURIBOR_3M
from util.exceptions import ValidationError


class TestSwaptionConstruction(unittest.TestCase):
    """Test Swaption construction."""

    def setUp(self):
        """Set up common test fixtures."""
        self.exercise_date = datetime(2025, 6, 15)
        self.swap_end_date = datetime(2030, 6, 15)
        self.notional = 50_000_000.0
        self.fixed_rate = 0.04

    def test_payer_swaption_construction(self):
        """Test basic payer swaption construction."""
        swaption = Swaption(
            exercise_date=self.exercise_date,
            swaption_type=SwaptionType.PAYER,
            swap_end_date=self.swap_end_date,
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            index=SOFR_3M,
        )

        self.assertEqual(swaption.swaption_type, SwaptionType.PAYER)
        self.assertEqual(swaption.exercise_date, self.exercise_date)
        self.assertEqual(swaption.swap_start_date, self.exercise_date)
        self.assertEqual(swaption.swap_end_date, self.swap_end_date)
        self.assertEqual(swaption.notional, self.notional)
        self.assertEqual(swaption.fixed_rate, self.fixed_rate)
        self.assertEqual(swaption.exercise_style, SwaptionExerciseStyle.EUROPEAN)

    def test_receiver_swaption_construction(self):
        """Test basic receiver swaption construction."""
        swaption = Swaption(
            exercise_date=self.exercise_date,
            swaption_type=SwaptionType.RECEIVER,
            swap_end_date=self.swap_end_date,
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            index=SOFR_3M,
        )

        self.assertEqual(swaption.swaption_type, SwaptionType.RECEIVER)

    def test_swap_start_defaults_to_exercise(self):
        """Test that swap_start_date defaults to exercise_date."""
        swaption = Swaption(
            exercise_date=self.exercise_date,
            swaption_type=SwaptionType.PAYER,
            swap_end_date=self.swap_end_date,
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            index=SOFR_3M,
        )
        self.assertEqual(swaption.swap_start_date, swaption.exercise_date)

    def test_forward_starting_swaption(self):
        """Test swaption with swap starting after exercise."""
        swap_start = datetime(2025, 9, 15)
        swaption = Swaption(
            exercise_date=self.exercise_date,
            swaption_type=SwaptionType.PAYER,
            swap_start_date=swap_start,
            swap_end_date=self.swap_end_date,
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            index=SOFR_3M,
        )
        self.assertEqual(swaption.swap_start_date, swap_start)


class TestSwaptionMethods(unittest.TestCase):
    """Test Swaption methods."""

    def setUp(self):
        self.swaption = Swaption(
            exercise_date=datetime(2025, 6, 15),
            swaption_type=SwaptionType.PAYER,
            swap_end_date=datetime(2030, 6, 15),
            notional=50_000_000,
            fixed_rate=0.04,
            index=SOFR_3M,
        )

    def test_get_notional(self):
        """Test get_notional returns the notional."""
        self.assertEqual(self.swaption.get_notional(), 50_000_000)

    def test_get_exercise_date(self):
        """Test get_exercise_date."""
        self.assertEqual(
            self.swaption.get_exercise_date(), datetime(2025, 6, 15)
        )

    def test_get_swap_tenor(self):
        """Test swap tenor calculation."""
        tenor = self.swaption.get_swap_tenor()
        self.assertAlmostEqual(tenor, 5.0, delta=0.02)

    def test_get_option_tenor(self):
        """Test option tenor from valuation date."""
        tenor = self.swaption.get_option_tenor(datetime(2024, 6, 15))
        self.assertAlmostEqual(tenor, 1.0, delta=0.02)

    def test_get_maturity(self):
        """Test maturity returns swap end date."""
        self.assertEqual(
            self.swaption.get_maturity(), datetime(2030, 6, 15)
        )

    def test_time_to_expiry(self):
        """Test time to expiry calculation."""
        t = self.swaption.time_to_expiry(datetime(2024, 6, 15))
        self.assertAlmostEqual(t, 1.0, delta=0.02)

    def test_is_expired(self):
        """Test expiry check."""
        self.assertFalse(self.swaption.is_expired(datetime(2024, 1, 1)))
        self.assertFalse(self.swaption.is_expired(datetime(2025, 6, 14)))
        self.assertTrue(self.swaption.is_expired(datetime(2025, 6, 15)))
        self.assertTrue(self.swaption.is_expired(datetime(2026, 1, 1)))

    def test_create_underlying_swap_payer(self):
        """Test creating underlying swap for payer swaption."""
        swap = self.swaption.create_underlying_swap()

        self.assertIsInstance(swap, InterestRateSwap)
        self.assertEqual(swap.direction, SwapDirection.PAYER)
        self.assertEqual(swap.get_fixed_rate(), 0.04)
        self.assertEqual(swap.get_start_date(), datetime(2025, 6, 15))
        self.assertEqual(swap.get_end_date(), datetime(2030, 6, 15))

    def test_create_underlying_swap_receiver(self):
        """Test creating underlying swap for receiver swaption."""
        swaption = Swaption(
            exercise_date=datetime(2025, 6, 15),
            swaption_type=SwaptionType.RECEIVER,
            swap_end_date=datetime(2030, 6, 15),
            notional=50_000_000,
            fixed_rate=0.04,
            index=SOFR_3M,
        )
        swap = swaption.create_underlying_swap()

        self.assertIsInstance(swap, InterestRateSwap)
        self.assertEqual(swap.direction, SwapDirection.RECEIVER)

    def test_repr(self):
        """Test string representation."""
        r = repr(self.swaption)
        self.assertIn("Swaption", r)
        self.assertIn("Payer", r)
        self.assertIn("SOFR_3M", r)


class TestSwaptionValidation(unittest.TestCase):
    """Test Swaption validation."""

    def test_negative_notional_raises(self):
        """Test negative notional raises."""
        with self.assertRaises(ValidationError):
            Swaption(
                exercise_date=datetime(2025, 6, 15),
                swaption_type=SwaptionType.PAYER,
                swap_end_date=datetime(2030, 6, 15),
                notional=-50_000_000,
                fixed_rate=0.04,
                index=SOFR_3M,
            )

    def test_unreasonable_rate_raises(self):
        """Test unreasonable rate raises."""
        with self.assertRaises(ValidationError):
            Swaption(
                exercise_date=datetime(2025, 6, 15),
                swaption_type=SwaptionType.PAYER,
                swap_end_date=datetime(2030, 6, 15),
                notional=50_000_000,
                fixed_rate=0.60,
                index=SOFR_3M,
            )

    def test_swap_end_before_start_raises(self):
        """Test swap end before start raises."""
        with self.assertRaises(ValidationError):
            Swaption(
                exercise_date=datetime(2025, 6, 15),
                swaption_type=SwaptionType.PAYER,
                swap_start_date=datetime(2025, 6, 15),
                swap_end_date=datetime(2024, 6, 15),
                notional=50_000_000,
                fixed_rate=0.04,
                index=SOFR_3M,
            )

    def test_swap_start_before_exercise_raises(self):
        """Test swap start before exercise date raises."""
        with self.assertRaises(ValidationError):
            Swaption(
                exercise_date=datetime(2025, 6, 15),
                swaption_type=SwaptionType.PAYER,
                swap_start_date=datetime(2025, 1, 15),
                swap_end_date=datetime(2030, 6, 15),
                notional=50_000_000,
                fixed_rate=0.04,
                index=SOFR_3M,
            )

    def test_trade_date_after_exercise_raises(self):
        """Test trade date after exercise date raises."""
        with self.assertRaises(ValidationError):
            Swaption(
                exercise_date=datetime(2025, 6, 15),
                swaption_type=SwaptionType.PAYER,
                swap_end_date=datetime(2030, 6, 15),
                notional=50_000_000,
                fixed_rate=0.04,
                index=SOFR_3M,
                trade_date=datetime(2025, 7, 1),
            )

    def test_no_index_raises(self):
        """Test missing index raises."""
        with self.assertRaises(ValidationError):
            Swaption(
                exercise_date=datetime(2025, 6, 15),
                swaption_type=SwaptionType.PAYER,
                swap_end_date=datetime(2030, 6, 15),
                notional=50_000_000,
                fixed_rate=0.04,
            )


class TestCreateSwaptionFactories(unittest.TestCase):
    """Test swaption factory functions."""

    def test_create_payer_swaption(self):
        """Test create_payer_swaption factory."""
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 6, 15),
            swap_tenor_years=5,
            notional=50_000_000,
            fixed_rate=0.04,
            index=SOFR_3M,
        )

        self.assertEqual(swaption.swaption_type, SwaptionType.PAYER)
        self.assertEqual(swaption.exercise_date, datetime(2025, 6, 15))
        self.assertEqual(swaption.swap_end_date, datetime(2030, 6, 15))
        self.assertAlmostEqual(swaption.get_swap_tenor(), 5.0, delta=0.02)

    def test_create_receiver_swaption(self):
        """Test create_receiver_swaption factory."""
        swaption = create_receiver_swaption(
            exercise_date=datetime(2025, 6, 15),
            swap_tenor_years=10,
            notional=100_000_000,
            fixed_rate=0.035,
            index=SOFR_3M,
        )

        self.assertEqual(swaption.swaption_type, SwaptionType.RECEIVER)
        self.assertEqual(swaption.swap_end_date, datetime(2035, 6, 15))

    def test_create_swaption_invalid_tenor(self):
        """Test that zero/negative tenor raises."""
        with self.assertRaises(ValidationError):
            create_payer_swaption(
                exercise_date=datetime(2025, 6, 15),
                swap_tenor_years=0,
                notional=50_000_000,
                fixed_rate=0.04,
                index=SOFR_3M,
            )

    def test_create_swaption_with_trade_date(self):
        """Test swaption with trade date."""
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 6, 15),
            swap_tenor_years=5,
            notional=50_000_000,
            fixed_rate=0.04,
            index=SOFR_3M,
            trade_date=datetime(2024, 1, 15),
        )
        self.assertEqual(swaption.trade_date, datetime(2024, 1, 15))

    def test_create_swaption_with_euribor(self):
        """Test swaption with EURIBOR index."""
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 6, 15),
            swap_tenor_years=5,
            notional=25_000_000,
            fixed_rate=0.03,
            index=EURIBOR_3M,
        )
        self.assertEqual(swaption.index, EURIBOR_3M)

    def test_create_swaption_bermudan(self):
        """Test swaption with Bermudan exercise style."""
        swaption = create_payer_swaption(
            exercise_date=datetime(2025, 6, 15),
            swap_tenor_years=5,
            notional=50_000_000,
            fixed_rate=0.04,
            index=SOFR_3M,
            exercise_style=SwaptionExerciseStyle.BERMUDAN,
        )
        self.assertEqual(
            swaption.exercise_style, SwaptionExerciseStyle.BERMUDAN
        )


if __name__ == "__main__":
    unittest.main()
