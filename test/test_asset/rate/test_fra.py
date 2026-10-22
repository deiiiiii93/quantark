"""
Tests for Forward Rate Agreement (FRA) product.
"""

import unittest
from datetime import datetime

from quantark.asset.rate.product.fra import ForwardRateAgreement, create_fra
from quantark.param.index import SOFR_3M, EURIBOR_3M
from quantark.util.calendar import DayCountConvention
from quantark.util.exceptions import ValidationError


class TestForwardRateAgreement(unittest.TestCase):
    """Test ForwardRateAgreement construction and methods."""

    def setUp(self):
        """Set up common test fixtures."""
        self.valuation_date = datetime(2024, 1, 15)
        self.accrual_start = datetime(2024, 4, 15)
        self.accrual_end = datetime(2024, 7, 15)
        self.notional = 10_000_000.0
        self.fixed_rate = 0.05

    def test_basic_construction(self):
        """Test basic FRA construction."""
        fra = ForwardRateAgreement(
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            accrual_start=self.accrual_start,
            accrual_end=self.accrual_end,
            index=SOFR_3M,
        )

        self.assertEqual(fra.notional, self.notional)
        self.assertEqual(fra.fixed_rate, self.fixed_rate)
        self.assertEqual(fra.accrual_start, self.accrual_start)
        self.assertEqual(fra.accrual_end, self.accrual_end)
        self.assertEqual(fra.index, SOFR_3M)
        self.assertEqual(fra.day_count_convention, DayCountConvention.ACT_360)

    def test_get_notional(self):
        """Test get_notional returns the notional."""
        fra = ForwardRateAgreement(
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            accrual_start=self.accrual_start,
            accrual_end=self.accrual_end,
            index=SOFR_3M,
        )
        self.assertEqual(fra.get_notional(), self.notional)

    def test_get_accrual_start(self):
        """Test accrual start getter."""
        fra = ForwardRateAgreement(
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            accrual_start=self.accrual_start,
            accrual_end=self.accrual_end,
            index=SOFR_3M,
        )
        self.assertEqual(fra.get_accrual_start(), self.accrual_start)

    def test_get_maturity(self):
        """Test maturity returns accrual end."""
        fra = ForwardRateAgreement(
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            accrual_start=self.accrual_start,
            accrual_end=self.accrual_end,
            index=SOFR_3M,
        )
        self.assertEqual(fra.get_maturity(), self.accrual_end)

    def test_time_to_settlement(self):
        """Test time to settlement calculation."""
        fra = ForwardRateAgreement(
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            accrual_start=self.accrual_start,
            accrual_end=self.accrual_end,
            index=SOFR_3M,
        )
        t = fra.time_to_settlement(self.valuation_date)
        expected = (self.accrual_start - self.valuation_date).days / 365.0
        self.assertAlmostEqual(t, expected, places=6)
        self.assertGreater(t, 0)

    def test_time_to_maturity(self):
        """Test time to maturity calculation."""
        fra = ForwardRateAgreement(
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            accrual_start=self.accrual_start,
            accrual_end=self.accrual_end,
            index=SOFR_3M,
        )
        t = fra.time_to_maturity(self.valuation_date)
        expected = (self.accrual_end - self.valuation_date).days / 365.0
        self.assertAlmostEqual(t, expected, places=6)

    def test_day_count_fraction(self):
        """Test day count fraction calculation."""
        fra = ForwardRateAgreement(
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            accrual_start=self.accrual_start,
            accrual_end=self.accrual_end,
            index=SOFR_3M,
            day_count_convention=DayCountConvention.ACT_360,
        )
        dcf = fra.day_count_fraction()
        # 91 days from Apr 15 to Jul 15
        expected_dcf = 91 / 360.0
        self.assertAlmostEqual(dcf, expected_dcf, places=4)

    def test_is_expired(self):
        """Test expiry check."""
        fra = ForwardRateAgreement(
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            accrual_start=self.accrual_start,
            accrual_end=self.accrual_end,
            index=SOFR_3M,
        )
        self.assertFalse(fra.is_expired(datetime(2024, 1, 15)))
        self.assertFalse(fra.is_expired(datetime(2024, 4, 14)))
        self.assertTrue(fra.is_expired(datetime(2024, 4, 15)))
        self.assertTrue(fra.is_expired(datetime(2024, 7, 15)))

    def test_repr(self):
        """Test string representation."""
        fra = ForwardRateAgreement(
            notional=self.notional,
            fixed_rate=self.fixed_rate,
            accrual_start=self.accrual_start,
            accrual_end=self.accrual_end,
            index=SOFR_3M,
        )
        r = repr(fra)
        self.assertIn("ForwardRateAgreement", r)
        self.assertIn("SOFR_3M", r)


class TestForwardRateAgreementValidation(unittest.TestCase):
    """Test FRA validation."""

    def test_negative_notional_raises(self):
        """Test that negative notional raises ValidationError."""
        with self.assertRaises(ValidationError):
            ForwardRateAgreement(
                notional=-1_000_000,
                fixed_rate=0.05,
                accrual_start=datetime(2024, 4, 15),
                accrual_end=datetime(2024, 7, 15),
                index=SOFR_3M,
            )

    def test_end_before_start_raises(self):
        """Test that accrual_end before accrual_start raises."""
        with self.assertRaises(ValidationError):
            ForwardRateAgreement(
                notional=10_000_000,
                fixed_rate=0.05,
                accrual_start=datetime(2024, 7, 15),
                accrual_end=datetime(2024, 4, 15),
                index=SOFR_3M,
            )

    def test_unreasonable_rate_raises(self):
        """Test that unreasonable rates raise."""
        with self.assertRaises(ValidationError):
            ForwardRateAgreement(
                notional=10_000_000,
                fixed_rate=0.60,
                accrual_start=datetime(2024, 4, 15),
                accrual_end=datetime(2024, 7, 15),
                index=SOFR_3M,
            )

    def test_trade_date_after_settlement_raises(self):
        """Test that trade date after settlement date raises."""
        with self.assertRaises(ValidationError):
            ForwardRateAgreement(
                notional=10_000_000,
                fixed_rate=0.05,
                accrual_start=datetime(2024, 4, 15),
                accrual_end=datetime(2024, 7, 15),
                index=SOFR_3M,
                trade_date=datetime(2024, 5, 1),
            )


class TestCreateFra(unittest.TestCase):
    """Test create_fra factory function."""

    def test_create_3m_fra(self):
        """Test creating a 3-month FRA."""
        fra = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 4, 15),
            tenor_months=3,
            notional=10_000_000,
            fixed_rate=0.05,
            index=SOFR_3M,
        )

        self.assertEqual(fra.notional, 10_000_000)
        self.assertEqual(fra.fixed_rate, 0.05)
        self.assertEqual(fra.accrual_start, datetime(2024, 4, 15))
        self.assertEqual(fra.accrual_end, datetime(2024, 7, 15))
        self.assertEqual(fra.trade_date, datetime(2024, 1, 15))

    def test_create_6m_fra(self):
        """Test creating a 6-month FRA."""
        fra = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 7, 15),
            tenor_months=6,
            notional=50_000_000,
            fixed_rate=0.045,
            index=EURIBOR_3M,
        )

        self.assertEqual(fra.accrual_start, datetime(2024, 7, 15))
        self.assertEqual(fra.accrual_end, datetime(2025, 1, 15))

    def test_create_fra_invalid_tenor(self):
        """Test that zero/negative tenor raises."""
        with self.assertRaises(ValidationError):
            create_fra(
                trade_date=datetime(2024, 1, 15),
                settlement_date=datetime(2024, 4, 15),
                tenor_months=0,
                notional=10_000_000,
                fixed_rate=0.05,
                index=SOFR_3M,
            )

    def test_create_fra_with_euribor(self):
        """Test FRA with EURIBOR index."""
        fra = create_fra(
            trade_date=datetime(2024, 1, 15),
            settlement_date=datetime(2024, 4, 15),
            tenor_months=3,
            notional=5_000_000,
            fixed_rate=0.035,
            index=EURIBOR_3M,
        )
        self.assertEqual(fra.index, EURIBOR_3M)


if __name__ == "__main__":
    unittest.main()
