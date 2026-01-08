"""
Comprehensive tests for convertible bond product.
"""
import unittest
from datetime import datetime

from asset.bond.product.convertible.convertible_bond import (
    ConvertibleBond,
    CallScheduleEntry,
    PutScheduleEntry,
    DiscreteDividend,
)
from util.calendar import DayCountConvention
from util.enum import PaymentFrequency
from util.exceptions import ValidationError


class TestCallScheduleEntry(unittest.TestCase):
    """Test CallScheduleEntry dataclass."""

    def test_simple_call_entry(self):
        """Test creating a simple call entry."""
        entry = CallScheduleEntry(
            call_date=datetime(2025, 1, 1),
            call_price=100.0,
        )
        self.assertEqual(entry.call_date, datetime(2025, 1, 1))
        self.assertEqual(entry.call_price, 100.0)
        self.assertFalse(entry.soft_call)
        self.assertIsNone(entry.trigger_level)

    def test_soft_call_entry(self):
        """Test creating a soft call entry."""
        entry = CallScheduleEntry(
            call_date=datetime(2025, 1, 1),
            call_price=100.0,
            soft_call=True,
            trigger_level=130.0,  # 130% of conversion price
        )
        self.assertTrue(entry.soft_call)
        self.assertEqual(entry.trigger_level, 130.0)

    def test_invalid_call_price(self):
        """Test that negative call price raises error."""
        with self.assertRaises(ValidationError):
            CallScheduleEntry(
                call_date=datetime(2025, 1, 1),
                call_price=-100.0,
            )

    def test_soft_call_without_trigger(self):
        """Test that soft call without trigger raises error."""
        with self.assertRaises(ValidationError):
            CallScheduleEntry(
                call_date=datetime(2025, 1, 1),
                call_price=100.0,
                soft_call=True,
                # Missing trigger_level
            )


class TestPutScheduleEntry(unittest.TestCase):
    """Test PutScheduleEntry dataclass."""

    def test_simple_put_entry(self):
        """Test creating a simple put entry."""
        entry = PutScheduleEntry(
            put_date=datetime(2026, 1, 1),
            put_price=100.0,
        )
        self.assertEqual(entry.put_date, datetime(2026, 1, 1))
        self.assertEqual(entry.put_price, 100.0)

    def test_invalid_put_price(self):
        """Test that negative put price raises error."""
        with self.assertRaises(ValidationError):
            PutScheduleEntry(
                put_date=datetime(2026, 1, 1),
                put_price=-100.0,
            )


class TestDiscreteDividend(unittest.TestCase):
    """Test DiscreteDividend dataclass."""

    def test_simple_dividend(self):
        """Test creating a simple dividend."""
        div = DiscreteDividend(
            ex_date=datetime(2024, 6, 15),
            amount=0.50,
        )
        self.assertEqual(div.ex_date, datetime(2024, 6, 15))
        self.assertEqual(div.amount, 0.50)

    def test_invalid_dividend_amount(self):
        """Test that negative dividend raises error."""
        with self.assertRaises(ValidationError):
            DiscreteDividend(
                ex_date=datetime(2024, 6, 15),
                amount=-0.50,
            )


class TestConvertibleBondCreation(unittest.TestCase):
    """Test ConvertibleBond creation."""

    def test_minimal_bond_creation(self):
        """Test creating a bond with minimal parameters."""
        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.02,
            conversion_ratio=10.0,
        )

        self.assertEqual(cb.face_value, 100.0)
        self.assertEqual(cb.coupon_rate, 0.02)
        self.assertEqual(cb.conversion_ratio, 10.0)
        self.assertEqual(cb.conversion_price, 10.0)  # 100 / 10
        self.assertEqual(cb.payment_frequency, PaymentFrequency.SEMI_ANNUAL)

    def test_conversion_price_calculation(self):
        """Test that conversion price is calculated from ratio."""
        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=1000.0,
            coupon_rate=0.03,
            conversion_ratio=25.0,
        )
        self.assertEqual(cb.conversion_price, 40.0)  # 1000 / 25

    def test_explicit_conversion_price(self):
        """Test providing explicit conversion price."""
        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.02,
            conversion_ratio=10.0,
            conversion_price=12.0,  # Different from face_value/ratio
        )
        self.assertEqual(cb.conversion_price, 12.0)

    def test_default_conversion_dates(self):
        """Test default conversion dates are set correctly."""
        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.02,
            conversion_ratio=10.0,
        )
        self.assertEqual(cb.conversion_start_date, datetime(2024, 1, 1))
        self.assertEqual(cb.conversion_end_date, datetime(2029, 1, 1))

    def test_full_bond_creation(self):
        """Test creating a fully specified bond."""
        call_schedule = [
            CallScheduleEntry(
                call_date=datetime(2026, 1, 1),
                call_price=103.0,
            ),
            CallScheduleEntry(
                call_date=datetime(2027, 1, 1),
                call_price=101.0,
            ),
        ]
        put_schedule = [
            PutScheduleEntry(
                put_date=datetime(2026, 6, 1),
                put_price=100.0,
            ),
        ]
        dividends = [
            DiscreteDividend(ex_date=datetime(2024, 6, 15), amount=0.50),
            DiscreteDividend(ex_date=datetime(2024, 12, 15), amount=0.50),
        ]

        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.025,
            conversion_ratio=10.0,
            payment_frequency=PaymentFrequency.QUARTERLY,
            day_count_convention=DayCountConvention.ACT_360,
            conversion_start_date=datetime(2024, 7, 1),
            conversion_end_date=datetime(2028, 7, 1),
            call_schedule=call_schedule,
            put_schedule=put_schedule,
            credit_spread=0.02,
            hazard_rate=0.01,
            recovery_rate=0.4,
            stock_jump_on_default=0.3,
            continuous_dividend_yield=0.01,
            discrete_dividends=dividends,
        )

        self.assertEqual(cb.payment_frequency, PaymentFrequency.QUARTERLY)
        self.assertEqual(len(cb.call_schedule), 2)
        self.assertEqual(len(cb.put_schedule), 1)
        self.assertEqual(len(cb.discrete_dividends), 2)
        self.assertEqual(cb.credit_spread, 0.02)


class TestConvertibleBondValidation(unittest.TestCase):
    """Test ConvertibleBond validation."""

    def test_invalid_dates(self):
        """Test that maturity before issue raises error."""
        with self.assertRaises(ValidationError):
            ConvertibleBond(
                issue_date=datetime(2029, 1, 1),
                maturity_date=datetime(2024, 1, 1),
                face_value=100.0,
                coupon_rate=0.02,
                conversion_ratio=10.0,
            )

    def test_invalid_conversion_dates(self):
        """Test that conversion start after end raises error."""
        with self.assertRaises(ValidationError):
            ConvertibleBond(
                issue_date=datetime(2024, 1, 1),
                maturity_date=datetime(2029, 1, 1),
                face_value=100.0,
                coupon_rate=0.02,
                conversion_ratio=10.0,
                conversion_start_date=datetime(2028, 1, 1),
                conversion_end_date=datetime(2025, 1, 1),
            )

    def test_invalid_face_value(self):
        """Test that non-positive face value raises error."""
        with self.assertRaises(ValidationError):
            ConvertibleBond(
                issue_date=datetime(2024, 1, 1),
                maturity_date=datetime(2029, 1, 1),
                face_value=0.0,
                coupon_rate=0.02,
                conversion_ratio=10.0,
            )

    def test_negative_coupon_rate(self):
        """Test that negative coupon raises error."""
        with self.assertRaises(ValidationError):
            ConvertibleBond(
                issue_date=datetime(2024, 1, 1),
                maturity_date=datetime(2029, 1, 1),
                face_value=100.0,
                coupon_rate=-0.02,
                conversion_ratio=10.0,
            )

    def test_negative_credit_spread(self):
        """Test that negative credit spread raises error."""
        with self.assertRaises(ValidationError):
            ConvertibleBond(
                issue_date=datetime(2024, 1, 1),
                maturity_date=datetime(2029, 1, 1),
                face_value=100.0,
                coupon_rate=0.02,
                conversion_ratio=10.0,
                credit_spread=-0.01,
            )

    def test_invalid_recovery_rate(self):
        """Test that recovery rate outside [0,1] raises error."""
        with self.assertRaises(ValidationError):
            ConvertibleBond(
                issue_date=datetime(2024, 1, 1),
                maturity_date=datetime(2029, 1, 1),
                face_value=100.0,
                coupon_rate=0.02,
                conversion_ratio=10.0,
                recovery_rate=1.5,
            )

    def test_invalid_call_schedule_order(self):
        """Test that out-of-order call schedule raises error."""
        call_schedule = [
            CallScheduleEntry(call_date=datetime(2027, 1, 1), call_price=101.0),
            CallScheduleEntry(call_date=datetime(2026, 1, 1), call_price=103.0),
        ]
        with self.assertRaises(ValidationError):
            ConvertibleBond(
                issue_date=datetime(2024, 1, 1),
                maturity_date=datetime(2029, 1, 1),
                face_value=100.0,
                coupon_rate=0.02,
                conversion_ratio=10.0,
                call_schedule=call_schedule,
            )


class TestConvertibleBondMethods(unittest.TestCase):
    """Test ConvertibleBond methods."""

    def setUp(self):
        """Set up test bond."""
        self.call_schedule = [
            CallScheduleEntry(
                call_date=datetime(2026, 1, 1),
                call_price=103.0,
            ),
            CallScheduleEntry(
                call_date=datetime(2027, 1, 1),
                call_price=101.0,
                soft_call=True,
                trigger_level=130.0,
            ),
        ]
        self.put_schedule = [
            PutScheduleEntry(
                put_date=datetime(2026, 6, 1),
                put_price=100.0,
            ),
        ]
        self.dividends = [
            DiscreteDividend(ex_date=datetime(2024, 6, 15), amount=0.50),
            DiscreteDividend(ex_date=datetime(2025, 6, 15), amount=0.55),
        ]

        self.cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.03,
            conversion_ratio=10.0,
            conversion_start_date=datetime(2024, 7, 1),
            conversion_end_date=datetime(2028, 7, 1),
            call_schedule=self.call_schedule,
            put_schedule=self.put_schedule,
            discrete_dividends=self.dividends,
        )

    def test_parity_calculation(self):
        """Test conversion parity calculation."""
        # At stock price 12, with conversion ratio 10
        parity = self.cb.parity(12.0)
        self.assertEqual(parity, 120.0)

    def test_conversion_premium(self):
        """Test conversion premium calculation."""
        # Parity = 10 * 12 = 120
        # Bond price = 130
        # Premium = (130 - 120) / 120 = 8.33%
        premium = self.cb.conversion_premium(12.0, 130.0)
        self.assertAlmostEqual(premium, 10.0 / 120.0, places=6)

    def test_is_callable_hard_call(self):
        """Test hard call detection."""
        # Before call date
        self.assertFalse(self.cb.is_callable_at(datetime(2025, 6, 1)))
        # After first hard call date
        self.assertTrue(self.cb.is_callable_at(datetime(2026, 6, 1)))

    def test_is_callable_soft_call(self):
        """Test soft call detection with stock price check."""
        # Create a bond with only soft call (no hard call before)
        cb_soft_only = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.03,
            conversion_ratio=10.0,  # conversion price = 10
            call_schedule=[
                CallScheduleEntry(
                    call_date=datetime(2026, 1, 1),
                    call_price=100.0,
                    soft_call=True,
                    trigger_level=130.0,  # 130% of conversion price = 13
                ),
            ],
        )
        # Before call date
        self.assertFalse(cb_soft_only.is_callable_at(datetime(2025, 6, 1)))
        # After soft call date, but stock below trigger (130% of 10 = 13)
        self.assertFalse(
            cb_soft_only.is_callable_at(datetime(2026, 6, 1), stock_price=11.0)
        )
        # Stock above trigger
        self.assertTrue(
            cb_soft_only.is_callable_at(datetime(2026, 6, 1), stock_price=15.0)
        )

    def test_get_call_price(self):
        """Test getting call price at dates."""
        self.assertIsNone(self.cb.get_call_price_at(datetime(2025, 6, 1)))
        self.assertEqual(
            self.cb.get_call_price_at(datetime(2026, 6, 1)), 103.0
        )
        self.assertEqual(
            self.cb.get_call_price_at(datetime(2027, 6, 1)), 101.0
        )

    def test_is_puttable(self):
        """Test put detection."""
        self.assertFalse(self.cb.is_puttable_at(datetime(2025, 6, 1)))
        self.assertTrue(self.cb.is_puttable_at(datetime(2026, 6, 1)))

    def test_get_put_price(self):
        """Test getting put price at dates."""
        self.assertIsNone(self.cb.get_put_price_at(datetime(2025, 6, 1)))
        self.assertEqual(
            self.cb.get_put_price_at(datetime(2026, 6, 1)), 100.0
        )

    def test_is_convertible(self):
        """Test conversion window detection."""
        # Before conversion start
        self.assertFalse(self.cb.is_convertible_at(datetime(2024, 6, 1)))
        # Within conversion window
        self.assertTrue(self.cb.is_convertible_at(datetime(2026, 1, 1)))
        # After conversion end
        self.assertFalse(self.cb.is_convertible_at(datetime(2029, 1, 1)))

    def test_get_discrete_dividends(self):
        """Test getting dividends in a date range."""
        divs = self.cb.get_discrete_dividends_between(
            datetime(2024, 1, 1), datetime(2024, 12, 31)
        )
        self.assertEqual(len(divs), 1)
        self.assertEqual(divs[0].amount, 0.50)

        divs = self.cb.get_discrete_dividends_between(
            datetime(2024, 1, 1), datetime(2026, 12, 31)
        )
        self.assertEqual(len(divs), 2)


class TestConvertibleBondCashflows(unittest.TestCase):
    """Test ConvertibleBond cashflow generation."""

    def test_cashflow_generation(self):
        """Test that cashflows are generated correctly."""
        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2026, 1, 1),  # 2 years
            face_value=100.0,
            coupon_rate=0.04,
            conversion_ratio=10.0,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        )

        cashflows = cb.get_all_cashflows()

        # Should have 4 payments (2 years * 2 per year)
        self.assertEqual(len(cashflows), 4)

        # Last payment should include principal
        last_cf = cashflows[-1]
        self.assertGreater(last_cf.amount, 100.0)

    def test_future_cashflows(self):
        """Test getting future cashflows from valuation date."""
        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2026, 1, 1),
            face_value=100.0,
            coupon_rate=0.04,
            conversion_ratio=10.0,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        )

        # All 4 cashflows are future relative to issue
        future_cfs = cb.get_cashflows(datetime(2024, 1, 1))
        self.assertEqual(len(future_cfs), 4)

        # After first payment
        future_cfs = cb.get_cashflows(datetime(2024, 8, 1))
        self.assertEqual(len(future_cfs), 3)

    def test_accrued_interest(self):
        """Test accrued interest calculation."""
        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.06,
            conversion_ratio=10.0,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        )

        # Halfway through first coupon period
        accrued = cb.calculate_accrued_interest(datetime(2024, 4, 1))

        # Should be positive and roughly half a coupon
        self.assertGreater(accrued, 0)
        self.assertLess(accrued, 3.0)  # Less than full semi-annual coupon

    def test_coupon_payment(self):
        """Test coupon payment calculation."""
        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.04,
            conversion_ratio=10.0,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        )

        # 4% annual / 2 payments = 2% = $2 on $100
        expected = 100.0 * 0.04 / 2
        self.assertEqual(cb.get_coupon_payment(), expected)


class TestConvertibleBondBaseBondMethods(unittest.TestCase):
    """Test BaseBondProduct interface methods."""

    def setUp(self):
        """Set up test bond."""
        self.cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.03,
            conversion_ratio=10.0,
        )

    def test_get_maturity_date(self):
        """Test get_maturity_date method."""
        self.assertEqual(self.cb.get_maturity_date(), datetime(2029, 1, 1))

    def test_get_issue_date(self):
        """Test get_issue_date method."""
        self.assertEqual(self.cb.get_issue_date(), datetime(2024, 1, 1))

    def test_get_denominator(self):
        """Test get_denominator method."""
        self.assertEqual(self.cb.get_denominator(), 100.0)

    def test_time_to_maturity(self):
        """Test time to maturity calculation."""
        ttm = self.cb.time_to_maturity(datetime(2027, 1, 1))
        # 2 years = 730/365 days
        self.assertAlmostEqual(ttm, 2.0, delta=0.01)

    def test_is_expired(self):
        """Test is_expired method."""
        self.assertFalse(self.cb.is_expired(datetime(2028, 1, 1)))
        self.assertTrue(self.cb.is_expired(datetime(2029, 1, 1)))

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.cb)
        self.assertIn("ConvertibleBond", repr_str)
        self.assertIn("2024-01-01", repr_str)
        self.assertIn("2029-01-01", repr_str)


class TestConvertibleBondEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_zero_coupon_convertible(self):
        """Test zero coupon convertible bond."""
        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.0,  # Zero coupon
            conversion_ratio=10.0,
        )

        self.assertEqual(cb.coupon_rate, 0.0)
        self.assertEqual(cb.get_coupon_payment(), 0.0)

        # Should only have principal cashflow at maturity
        cashflows = cb.get_all_cashflows()
        # With frequency generator, might still have multiple "periods"
        # but amounts should be 0 except final principal
        for cf in cashflows[:-1]:
            self.assertAlmostEqual(cf.amount, 0.0, delta=0.01)

    def test_conversion_premium_at_parity(self):
        """Test conversion premium when bond trades at parity."""
        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.02,
            conversion_ratio=10.0,
        )

        # Bond trades at parity
        premium = cb.conversion_premium(10.0, 100.0)  # parity = 100
        self.assertAlmostEqual(premium, 0.0, places=6)

    def test_conversion_premium_when_stock_zero(self):
        """Test conversion premium handles zero stock price."""
        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.02,
            conversion_ratio=10.0,
        )

        premium = cb.conversion_premium(0.0, 100.0)
        self.assertEqual(premium, float("inf"))


if __name__ == "__main__":
    unittest.main()
