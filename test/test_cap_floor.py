"""
Tests for Interest Rate Cap, Floor, and Collar products.
"""

import unittest
from datetime import datetime

from quantark.asset.rate.product.cap_floor import (
    CapFloor,
    CapFloorType,
    Caplet,
    Collar,
    create_cap,
    create_floor,
    create_collar,
)
from quantark.asset.rate.product.irs import NotionalSchedule
from quantark.param.index import SOFR_3M, EURIBOR_3M, IndexFixingStore, IndexFixing
from quantark.util.enum import PaymentFrequency
from quantark.util.exceptions import ValidationError


class TestCapFloorConstruction(unittest.TestCase):
    """Test Cap/Floor construction."""

    def setUp(self):
        """Set up common test fixtures."""
        self.start_date = datetime(2024, 3, 15)
        self.end_date = datetime(2026, 3, 15)
        self.notional = 10_000_000.0
        self.strike = 0.05

    def test_cap_construction(self):
        """Test basic cap construction."""
        cap = CapFloor(
            notional=self.notional,
            strike=self.strike,
            cap_floor_type=CapFloorType.CAP,
            start_date=self.start_date,
            end_date=self.end_date,
            index=SOFR_3M,
        )

        self.assertEqual(cap.notional, self.notional)
        self.assertEqual(cap.strike, self.strike)
        self.assertEqual(cap.cap_floor_type, CapFloorType.CAP)
        self.assertEqual(cap.index, SOFR_3M)

    def test_floor_construction(self):
        """Test basic floor construction."""
        floor = CapFloor(
            notional=self.notional,
            strike=0.03,
            cap_floor_type=CapFloorType.FLOOR,
            start_date=self.start_date,
            end_date=self.end_date,
            index=SOFR_3M,
        )

        self.assertEqual(floor.cap_floor_type, CapFloorType.FLOOR)
        self.assertEqual(floor.strike, 0.03)

    def test_quarterly_caplets_count(self):
        """Test that quarterly cap over 2 years generates 8 caplets."""
        cap = create_cap(
            start_date=self.start_date,
            end_date=self.end_date,
            notional=self.notional,
            strike=self.strike,
            index=SOFR_3M,
            payment_frequency=PaymentFrequency.QUARTERLY,
        )

        self.assertEqual(cap.num_periods(), 8)

    def test_semi_annual_caplets_count(self):
        """Test that semi-annual cap over 2 years generates 4 caplets."""
        cap = create_cap(
            start_date=self.start_date,
            end_date=self.end_date,
            notional=self.notional,
            strike=self.strike,
            index=SOFR_3M,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        )

        self.assertEqual(cap.num_periods(), 4)

    def test_caplet_structure(self):
        """Test that caplets have correct structure."""
        cap = create_cap(
            start_date=self.start_date,
            end_date=self.end_date,
            notional=self.notional,
            strike=self.strike,
            index=SOFR_3M,
        )

        caplets = cap.get_caplets()
        self.assertTrue(len(caplets) > 0)

        for caplet in caplets:
            self.assertIsInstance(caplet, Caplet)
            self.assertGreater(caplet.accrual_end, caplet.accrual_start)
            self.assertEqual(caplet.notional, self.notional)
            self.assertEqual(caplet.strike, self.strike)
            self.assertGreater(caplet.day_count_fraction, 0)
            self.assertTrue(caplet.is_projected)

        # Caplets should be contiguous
        for i in range(1, len(caplets)):
            self.assertEqual(caplets[i].accrual_start, caplets[i - 1].accrual_end)

    def test_get_floorlets_alias(self):
        """Test that get_floorlets is alias for get_caplets."""
        floor = create_floor(
            start_date=self.start_date,
            end_date=self.end_date,
            notional=self.notional,
            strike=0.03,
            index=SOFR_3M,
        )

        caplets = floor.get_caplets()
        floorlets = floor.get_floorlets()
        self.assertEqual(len(caplets), len(floorlets))

    def test_get_future_caplets(self):
        """Test filtering caplets by valuation date."""
        cap = create_cap(
            start_date=self.start_date,
            end_date=self.end_date,
            notional=self.notional,
            strike=self.strike,
            index=SOFR_3M,
        )

        all_caplets = cap.get_caplets()
        future = cap.get_future_caplets(datetime(2025, 3, 15))
        self.assertLess(len(future), len(all_caplets))


class TestCapFloorMethods(unittest.TestCase):
    """Test Cap/Floor methods."""

    def test_get_notional(self):
        """Test get_notional returns constant notional."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        self.assertEqual(cap.get_notional(datetime(2025, 1, 1)), 10_000_000)

    def test_get_notional_amortizing(self):
        """Test amortizing notional schedule."""
        schedule = NotionalSchedule(
            notional_dates=[datetime(2025, 3, 15)],
            notional_amounts=[5_000_000],
            initial_notional=10_000_000,
        )

        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
            notional_schedule=schedule,
        )

        self.assertEqual(cap.get_notional(datetime(2024, 6, 1)), 10_000_000)
        self.assertEqual(cap.get_notional(datetime(2025, 6, 1)), 5_000_000)

    def test_get_maturity(self):
        """Test maturity getter."""
        end = datetime(2026, 3, 15)
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=end,
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        self.assertEqual(cap.get_maturity(), end)

    def test_time_to_maturity(self):
        """Test time to maturity calculation."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        t = cap.time_to_maturity(datetime(2024, 3, 15))
        self.assertAlmostEqual(t, 2.0, delta=0.02)

    def test_is_expired(self):
        """Test expiry check."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        self.assertFalse(cap.is_expired(datetime(2025, 1, 1)))
        self.assertTrue(cap.is_expired(datetime(2026, 3, 15)))
        self.assertTrue(cap.is_expired(datetime(2027, 1, 1)))

    def test_repr_cap(self):
        """Test cap string representation."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        r = repr(cap)
        self.assertIn("CAP", r)
        self.assertIn("SOFR_3M", r)

    def test_repr_floor(self):
        """Test floor string representation."""
        floor = create_floor(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.03,
            index=SOFR_3M,
        )
        r = repr(floor)
        self.assertIn("FLOOR", r)


class TestCapFloorValidation(unittest.TestCase):
    """Test Cap/Floor validation."""

    def test_negative_notional_raises(self):
        """Test negative notional raises."""
        with self.assertRaises(ValidationError):
            CapFloor(
                notional=-10_000_000,
                strike=0.05,
                cap_floor_type=CapFloorType.CAP,
                start_date=datetime(2024, 3, 15),
                end_date=datetime(2026, 3, 15),
                index=SOFR_3M,
            )

    def test_end_before_start_raises(self):
        """Test end before start raises."""
        with self.assertRaises(ValidationError):
            CapFloor(
                notional=10_000_000,
                strike=0.05,
                cap_floor_type=CapFloorType.CAP,
                start_date=datetime(2026, 3, 15),
                end_date=datetime(2024, 3, 15),
                index=SOFR_3M,
            )

    def test_unreasonable_strike_raises(self):
        """Test unreasonable strike raises."""
        with self.assertRaises(ValidationError):
            CapFloor(
                notional=10_000_000,
                strike=0.60,
                cap_floor_type=CapFloorType.CAP,
                start_date=datetime(2024, 3, 15),
                end_date=datetime(2026, 3, 15),
                index=SOFR_3M,
            )


class TestCapFloorWithFixings(unittest.TestCase):
    """Test Cap/Floor with historical fixings."""

    def test_caplet_with_fixing(self):
        """Test that caplets with historical fixings are marked correctly."""
        store = IndexFixingStore()
        store.add_fixing(IndexFixing(
            fixing_date=datetime(2024, 3, 15),
            rate=0.048,
            index_name="SOFR_3M",
        ))

        cap = CapFloor(
            notional=10_000_000,
            strike=0.05,
            cap_floor_type=CapFloorType.CAP,
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2025, 3, 15),
            index=SOFR_3M,
            fixing_store=store,
        )

        caplets = cap.get_caplets()
        # First caplet should have a fixing (fixing date near start)
        first = caplets[0]
        if not first.is_projected:
            self.assertAlmostEqual(first.index_fixing, 0.048, places=4)


class TestCollar(unittest.TestCase):
    """Test Collar construction and validation."""

    def test_collar_construction(self):
        """Test basic collar construction."""
        collar = create_collar(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            cap_strike=0.06,
            floor_strike=0.03,
            index=SOFR_3M,
        )

        self.assertIsInstance(collar, Collar)
        self.assertEqual(collar.cap.cap_floor_type, CapFloorType.CAP)
        self.assertEqual(collar.floor.cap_floor_type, CapFloorType.FLOOR)
        self.assertEqual(collar.cap.strike, 0.06)
        self.assertEqual(collar.floor.strike, 0.03)

    def test_collar_cap_above_floor(self):
        """Test that collar requires cap strike > floor strike."""
        with self.assertRaises(ValidationError):
            create_collar(
                start_date=datetime(2024, 3, 15),
                end_date=datetime(2026, 3, 15),
                notional=10_000_000,
                cap_strike=0.03,
                floor_strike=0.06,
                index=SOFR_3M,
            )

    def test_collar_equal_strikes_raises(self):
        """Test that equal strikes raise."""
        with self.assertRaises(ValidationError):
            create_collar(
                start_date=datetime(2024, 3, 15),
                end_date=datetime(2026, 3, 15),
                notional=10_000_000,
                cap_strike=0.05,
                floor_strike=0.05,
                index=SOFR_3M,
            )

    def test_collar_maturity(self):
        """Test collar maturity methods."""
        collar = create_collar(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            cap_strike=0.06,
            floor_strike=0.03,
            index=SOFR_3M,
        )

        self.assertEqual(collar.get_maturity(), datetime(2026, 3, 15))
        self.assertFalse(collar.is_expired(datetime(2025, 1, 1)))
        self.assertTrue(collar.is_expired(datetime(2026, 3, 15)))

    def test_collar_repr(self):
        """Test collar string representation."""
        collar = create_collar(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            cap_strike=0.06,
            floor_strike=0.03,
            index=SOFR_3M,
        )
        r = repr(collar)
        self.assertIn("Collar", r)


class TestCreateCapFloorFactories(unittest.TestCase):
    """Test factory functions."""

    def test_create_cap(self):
        """Test create_cap factory."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.05,
            index=SOFR_3M,
        )
        self.assertEqual(cap.cap_floor_type, CapFloorType.CAP)

    def test_create_floor(self):
        """Test create_floor factory."""
        floor = create_floor(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=10_000_000,
            strike=0.03,
            index=SOFR_3M,
        )
        self.assertEqual(floor.cap_floor_type, CapFloorType.FLOOR)

    def test_create_cap_with_euribor(self):
        """Test cap with EURIBOR index."""
        cap = create_cap(
            start_date=datetime(2024, 3, 15),
            end_date=datetime(2026, 3, 15),
            notional=5_000_000,
            strike=0.04,
            index=EURIBOR_3M,
        )
        self.assertEqual(cap.index, EURIBOR_3M)


if __name__ == "__main__":
    unittest.main()
