"""
Comprehensive unit tests for PhoenixOption.

Tests cover:
- Basic product creation (standard and reverse)
- Coupon barrier configuration and triggering
- Memory coupon feature
- Day count convention integration
- KO/KI barrier triggering
- Payoff calculations (V0 and V1 states)
- Validation errors
- Helper functions
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from util.exceptions import ValidationError
from asset.equity.product.option.phoenix_config import CouponBarrierConfig
from asset.equity.product.option.phoenix_option import PhoenixOption
from asset.equity.product.option.phoenix_helpers import (
    create_standard_phoenix,
    create_stepdown_phoenix,
    create_reverse_phoenix,
    create_memory_phoenix,
    create_non_memory_phoenix,
)
from asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from asset.equity.product.option.snowball_config import (
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from util.calendar.day_counter import DayCountConvention
from util.enum import (
    BarrierType,
    CouponPayType,
    ExerciseType,
    ObservationType,
    OptionType,
    ProtectionType,
)
from util.exceptions import ValidationError


# =============================================================================
# Fixtures - Common test configurations
# =============================================================================


def create_basic_barrier_config(
    ko_barrier: float = 103.0,
    ko_rate: float = 0.15,
    ki_barrier: float = 75.0,
    ko_observation_dates: List[float] = None,
) -> BarrierConfig:
    """Create a basic barrier configuration for testing."""
    if ko_observation_dates is None:
        ko_observation_dates = [0.25, 0.5, 0.75, 1.0]

    return BarrierConfig(
        ko_barrier=ko_barrier,
        ko_rate=ko_rate,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=ko_observation_dates,
        ki_barrier=ki_barrier,
        ki_observation_type=ObservationType.CONTINUOUS,
    )


def create_basic_coupon_config(
    coupon_barrier: float = 85.0,
    coupon_rate: float = 0.01,
    memory_coupon: bool = True,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_365,
    coupon_pay_type: CouponPayType = CouponPayType.INSTANT,
) -> CouponBarrierConfig:
    """Create a basic coupon barrier configuration for testing."""
    return CouponBarrierConfig(
        coupon_barrier=coupon_barrier,
        coupon_rate=coupon_rate,
        memory_coupon=memory_coupon,
        day_count_convention=day_count_convention,
        coupon_pay_type=coupon_pay_type,
    )


def create_test_phoenix(
    initial_price: float = 100.0,
    strike: float = 100.0,
    contract_multiplier: float = None,
    maturity: float = 1.0,
    barrier_config: BarrierConfig = None,
    coupon_config: CouponBarrierConfig = None,
    payoff_config: PayoffConfig = None,
    accrual_config: AccrualConfig = None,
    is_reverse: bool = False,
) -> PhoenixOption:
    """Create a phoenix option for testing."""
    if barrier_config is None:
        barrier_config = create_basic_barrier_config()
    if coupon_config is None:
        coupon_config = create_basic_coupon_config()
    if contract_multiplier is None:
        contract_multiplier = 1.0

    return PhoenixOption(
        initial_price=initial_price,
        strike=strike,
        barrier_config=barrier_config,
        coupon_config=coupon_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        contract_multiplier=contract_multiplier,
        maturity=maturity,
        is_reverse=is_reverse,
    )


def get_principal(phoenix: PhoenixOption) -> float:
    """Return per-contract principal based on initial price and contract multiplier."""
    return phoenix.initial_price * phoenix.contract_multiplier


# =============================================================================
# Test Classes
# =============================================================================


class TestPhoenixCreation:
    """Tests for basic phoenix option creation."""

    def test_standard_phoenix_creation(self):
        """Test basic standard phoenix creation."""
        phoenix = create_test_phoenix()

        assert phoenix.initial_price == 100.0
        assert phoenix.strike == 100.0
        assert get_principal(phoenix) == 100.0
        assert phoenix.maturity == 1.0
        assert phoenix.is_reverse is False
        assert phoenix.is_standard is True

    def test_reverse_phoenix_creation(self):
        """Test basic reverse phoenix creation."""
        phoenix = create_test_phoenix(is_reverse=True)

        assert phoenix.initial_price == 100.0
        assert phoenix.strike == 100.0
        assert phoenix.is_reverse is True
        assert phoenix.is_standard is False

    def test_option_type_standard(self):
        """Test standard phoenix has PUT option type."""
        phoenix = create_test_phoenix(is_reverse=False)
        assert phoenix.option_type == OptionType.PUT

    def test_option_type_reverse(self):
        """Test reverse phoenix has CALL option type."""
        phoenix = create_test_phoenix(is_reverse=True)
        assert phoenix.option_type == OptionType.CALL

    def test_exercise_type_european(self):
        """Test phoenix has European exercise type."""
        phoenix = create_test_phoenix()
        assert phoenix.exercise_type == ExerciseType.EUROPEAN


class TestCouponBarrierConfig:
    """Tests for CouponBarrierConfig class."""

    def test_basic_creation(self):
        """Test basic coupon config creation."""
        config = CouponBarrierConfig(
            coupon_barrier=85.0,
            coupon_rate=0.01,
        )
        assert config.coupon_barrier == 85.0
        assert config.coupon_rate == 0.01
        assert config.memory_coupon is True  # Default
        assert config.day_count_convention == DayCountConvention.ACT_365  # Default
        assert config.coupon_pay_type == CouponPayType.INSTANT  # Default

    def test_time_varying_barrier(self):
        """Test coupon config with time-varying barrier."""
        config = CouponBarrierConfig(
            coupon_barrier=[85.0, 84.0, 83.0, 82.0],
            coupon_rate=0.01,
        )
        assert isinstance(config.coupon_barrier, list)
        assert len(config.coupon_barrier) == 4

    def test_thirty_360_convention(self):
        """Test coupon config with 30/360 day count convention."""
        config = CouponBarrierConfig(
            coupon_barrier=85.0,
            coupon_rate=0.01,
            day_count_convention=DayCountConvention.THIRTY_360_US,
        )
        assert config.day_count_convention == DayCountConvention.THIRTY_360_US

    def test_coupon_pay_type_expiry(self):
        """Test coupon config with EXPIRY payment type."""
        config = CouponBarrierConfig(
            coupon_barrier=85.0,
            coupon_rate=0.01,
            coupon_pay_type=CouponPayType.EXPIRY,
        )
        assert config.coupon_pay_type == CouponPayType.EXPIRY

    def test_memory_coupon_disabled(self):
        """Test coupon config with memory coupon disabled."""
        config = CouponBarrierConfig(
            coupon_barrier=85.0,
            coupon_rate=0.01,
            memory_coupon=False,
        )
        assert config.memory_coupon is False

    def test_validation_negative_barrier(self):
        """Test validation error for negative coupon barrier."""
        with pytest.raises(ValidationError, match="coupon_barrier"):
            CouponBarrierConfig(coupon_barrier=-85.0, coupon_rate=0.01)

    def test_validation_zero_barrier(self):
        """Test validation error for zero coupon barrier."""
        with pytest.raises(ValidationError, match="coupon_barrier"):
            CouponBarrierConfig(coupon_barrier=0.0, coupon_rate=0.01)

    def test_validation_negative_rate(self):
        """Test validation error for negative coupon rate."""
        with pytest.raises(ValidationError, match="coupon_rate"):
            CouponBarrierConfig(coupon_barrier=85.0, coupon_rate=-0.01)


class TestCouponBarrierTriggering:
    """Tests for coupon barrier triggering logic."""

    def test_coupon_triggered_standard(self):
        """Test coupon triggers when spot >= barrier (standard)."""
        phoenix = create_test_phoenix()  # coupon_barrier=85

        assert phoenix.is_coupon_triggered(spot=90.0) is True
        assert phoenix.is_coupon_triggered(spot=85.0) is True
        assert phoenix.is_coupon_triggered(spot=80.0) is False

    def test_coupon_triggered_reverse(self):
        """Test coupon triggers when spot <= barrier (reverse)."""
        coupon_config = create_basic_coupon_config(coupon_barrier=115.0)
        phoenix = create_test_phoenix(coupon_config=coupon_config, is_reverse=True)

        assert phoenix.is_coupon_triggered(spot=110.0) is True
        assert phoenix.is_coupon_triggered(spot=115.0) is True
        assert phoenix.is_coupon_triggered(spot=120.0) is False

    def test_coupon_triggered_time_varying(self):
        """Test coupon triggering with time-varying barrier."""
        coupon_config = CouponBarrierConfig(
            coupon_barrier=[85.0, 84.0, 83.0, 82.0],
            coupon_rate=0.01,
        )
        phoenix = create_test_phoenix(coupon_config=coupon_config)

        # At obs 0, barrier is 85
        assert phoenix.is_coupon_triggered(spot=85.0, observation_idx=0) is True
        assert phoenix.is_coupon_triggered(spot=84.5, observation_idx=0) is False

        # At obs 2, barrier is 83
        assert phoenix.is_coupon_triggered(spot=83.0, observation_idx=2) is True
        assert phoenix.is_coupon_triggered(spot=82.5, observation_idx=2) is False

    def test_get_coupon_barrier_at(self):
        """Test getting coupon barrier at specific observation index."""
        coupon_config = CouponBarrierConfig(
            coupon_barrier=[85.0, 84.0, 83.0, 82.0],
            coupon_rate=0.01,
        )
        phoenix = create_test_phoenix(coupon_config=coupon_config)

        assert phoenix.get_coupon_barrier_at(0) == 85.0
        assert phoenix.get_coupon_barrier_at(1) == 84.0
        assert phoenix.get_coupon_barrier_at(2) == 83.0
        assert phoenix.get_coupon_barrier_at(3) == 82.0


class TestCouponPayoff:
    """Tests for coupon payoff calculation."""

    def test_coupon_payoff_basic(self):
        """Test basic coupon payoff calculation."""
        phoenix = create_test_phoenix()
        principal = get_principal(phoenix)

        # With default year_fraction (1.0)
        payoff = phoenix.get_coupon_payoff(observation_idx=0, year_fraction=1.0)
        expected = principal * 0.01 * 1.0  # 10,000
        assert payoff == expected

    def test_coupon_payoff_with_year_fraction(self):
        """Test coupon payoff with specific year fraction."""
        phoenix = create_test_phoenix()

        # 3 months = 0.25 year fraction
        payoff = phoenix.get_coupon_payoff(observation_idx=0, year_fraction=0.25)
        expected = get_principal(phoenix) * 0.01 * 0.25  # 2,500
        assert payoff == expected

    def test_fixed_coupon_year_fraction(self):
        """Test fixed per-period accrual fraction for equal monthly coupons."""
        coupon_config = CouponBarrierConfig(
            coupon_barrier=85.0,
            coupon_rate=0.12,
            fixed_coupon_year_fraction=1.0 / 12.0,
        )
        phoenix = create_test_phoenix(coupon_config=coupon_config)
        yfs = phoenix.get_coupon_period_year_fractions([0.1, 0.2, 0.35])
        assert yfs == [1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0]

    def test_external_accrual_factors_override_coupon_periods(self):
        """Test external accrual factors drive Phoenix periodic coupons."""
        coupon_config = CouponBarrierConfig(
            coupon_barrier=85.0,
            coupon_rate=0.12,
            fixed_coupon_year_fraction=1.0 / 12.0,
        )
        accrual_config = AccrualConfig(accrual_factors=[0.20, 0.35, 0.45, 0.60])
        phoenix = create_test_phoenix(
            coupon_config=coupon_config,
            accrual_config=accrual_config,
        )

        yfs = phoenix.get_coupon_period_year_fractions([0.1, 0.2, 0.35, 0.5])

        assert yfs == [0.20, 0.35, 0.45, 0.60]

    def test_coupon_payoff_with_dates(self):
        """Test coupon payoff calculation with dates."""
        phoenix = create_test_phoenix()

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 4, 1)  # 91 days

        payoff = phoenix.get_coupon_payoff(
            observation_idx=0,
            start_date=start_date,
            end_date=end_date,
        )
        # ACT/365: 91/365 = 0.2493...
        expected_dcf = 91 / 365
        expected = get_principal(phoenix) * 0.01 * expected_dcf
        assert abs(payoff - expected) < 0.01


class TestKOKITriggering:
    """Tests for KO/KI barrier triggering logic."""

    def test_ko_triggered_standard(self):
        """Test KO triggers when spot >= barrier (standard)."""
        phoenix = create_test_phoenix()  # ko_barrier=103

        assert phoenix.is_ko_triggered(spot=105.0) is True
        assert phoenix.is_ko_triggered(spot=103.0) is True
        assert phoenix.is_ko_triggered(spot=100.0) is False

    def test_ko_triggered_reverse(self):
        """Test KO triggers when spot <= barrier (reverse)."""
        barrier_config = create_basic_barrier_config(ko_barrier=97.0)
        phoenix = create_test_phoenix(barrier_config=barrier_config, is_reverse=True)

        assert phoenix.is_ko_triggered(spot=95.0) is True
        assert phoenix.is_ko_triggered(spot=97.0) is True
        assert phoenix.is_ko_triggered(spot=100.0) is False

    def test_ki_triggered_standard(self):
        """Test KI triggers when spot <= barrier (standard)."""
        phoenix = create_test_phoenix()  # ki_barrier=75

        assert phoenix.is_ki_triggered(spot=70.0) is True
        assert phoenix.is_ki_triggered(spot=75.0) is True
        assert phoenix.is_ki_triggered(spot=80.0) is False

    def test_ki_triggered_reverse(self):
        """Test KI triggers when spot >= barrier (reverse)."""
        barrier_config = create_basic_barrier_config(ki_barrier=125.0)
        phoenix = create_test_phoenix(barrier_config=barrier_config, is_reverse=True)

        assert phoenix.is_ki_triggered(spot=130.0) is True
        assert phoenix.is_ki_triggered(spot=125.0) is True
        assert phoenix.is_ki_triggered(spot=120.0) is False

    def test_no_ki_barrier(self):
        """Test KI triggering returns False when no KI barrier."""
        barrier_config = BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=None,  # No KI barrier
        )
        phoenix = create_test_phoenix(barrier_config=barrier_config)

        assert phoenix.is_ki_triggered(spot=50.0) is False
        assert phoenix.has_ki_barrier is False


class TestMaturityPayoffs:
    """Tests for maturity payoff calculations."""

    def test_v0_payoff_basic(self):
        """Test V0 payoff (not knocked-in)."""
        payoff_config = PayoffConfig(
            rebate_rate=0.15,
            include_principal=False,
        )
        phoenix = create_test_phoenix(payoff_config=payoff_config)

        # V0 payoff = rebate + accumulated coupons
        principal = get_principal(phoenix)
        accumulated = principal * 0.03
        payoff = phoenix.get_maturity_payoff_v0(
            spot=100.0, accumulated_coupons=accumulated
        )
        rebate = principal * 0.15
        expected = rebate + accumulated
        assert payoff == expected

    def test_v0_payoff_with_principal(self):
        """Test V0 payoff with principal included."""
        payoff_config = PayoffConfig(
            rebate_rate=0.15,
            include_principal=True,
        )
        phoenix = create_test_phoenix(payoff_config=payoff_config)

        payoff = phoenix.get_maturity_payoff_v0(spot=100.0, accumulated_coupons=0)
        principal = get_principal(phoenix)
        rebate = principal * 0.15
        expected = principal + rebate
        assert payoff == expected

    def test_v1_payoff_standard(self):
        """Test V1 payoff (knocked-in) for standard phoenix."""
        payoff_config = PayoffConfig(
            include_principal=True,
            participation_rate=1.0,
        )
        phoenix = create_test_phoenix(payoff_config=payoff_config)

        # Spot at 80, strike at 100 -> loss of 20%
        payoff = phoenix.get_maturity_payoff_v1(spot=80.0)
        principal = get_principal(phoenix)
        downside = -20.0 * phoenix.contract_multiplier
        expected = principal + downside
        assert payoff == expected

    def test_v1_payoff_reverse(self):
        """Test V1 payoff (knocked-in) for reverse phoenix."""
        payoff_config = PayoffConfig(
            include_principal=True,
            participation_rate=1.0,
        )
        barrier_config = create_basic_barrier_config(ki_barrier=125.0)
        phoenix = create_test_phoenix(
            barrier_config=barrier_config,
            payoff_config=payoff_config,
            is_reverse=True,
        )

        # Spot at 120, strike at 100 -> loss of 20% for reverse
        payoff = phoenix.get_maturity_payoff_v1(spot=120.0)
        principal = get_principal(phoenix)
        downside = -20.0 * phoenix.contract_multiplier
        expected = principal + downside
        assert payoff == expected

    def test_v1_payoff_with_protection(self):
        """Test V1 payoff with partial protection."""
        payoff_config = PayoffConfig(
            include_principal=True,
            participation_rate=1.0,
            protection_type=ProtectionType.PARTIAL,
            protection_rate=0.5,  # 50% protection
        )
        phoenix = create_test_phoenix(payoff_config=payoff_config)

        # Spot at 40, strike at 100 -> huge loss, but protected
        payoff = phoenix.get_maturity_payoff_v1(spot=40.0)
        principal = get_principal(phoenix)
        floor = 0.5 * phoenix.initial_price * phoenix.contract_multiplier
        expected = principal - floor
        assert payoff == expected


class TestKOPayoff:
    """Tests for KO payoff calculation."""

    def test_ko_payoff_with_coupons(self):
        """Test KO payoff includes accumulated coupons and current coupon."""
        # Use explicit ko_rate to test rebate logic, even though default is now 0.0
        barrier_config = create_basic_barrier_config(ko_rate=0.15)
        payoff_config = PayoffConfig(include_principal=False)
        phoenix = create_test_phoenix(
            barrier_config=barrier_config,
            payoff_config=payoff_config
        )

        accumulated = get_principal(phoenix) * 0.03
        payoff = phoenix.get_ko_payoff(
            spot=105.0,
            observation_idx=1,  # 0.5 years
            accumulated_coupons=accumulated,
        )
        
        # Components:
        # 1. KO rebate (Snowball-like bonus): 15% rate at 0.5 years
        ko_rate = 0.15
        accrual_factor = 0.5  # observation at 0.5
        ko_rebate = get_principal(phoenix) * ko_rate * accrual_factor
        
        # 2. Accumulated coupons (passed in)
        
        # 3. Current period coupon: Triggered because spot 105 >= coupon_barrier 85
        # Default coupon rate 1% per period (not annualized in default get_coupon_payoff)
        current_coupon = get_principal(phoenix) * 0.01 * 1.0
        
        expected = ko_rebate + accumulated + current_coupon
        assert payoff == expected

    def test_external_accrual_factors_drive_ko_and_current_coupon(self):
        """Test external accrual factors drive Phoenix KO and current coupon payoffs."""
        barrier_config = create_basic_barrier_config(ko_rate=0.15)
        payoff_config = PayoffConfig(include_principal=False)
        accrual_config = AccrualConfig(
            is_annualized=True,
            is_annualized_ko=True,
            accrual_factors=[0.10, 0.30, 0.60, 1.00],
        )
        phoenix = create_test_phoenix(
            barrier_config=barrier_config,
            payoff_config=payoff_config,
            accrual_config=accrual_config,
        )

        payoff = phoenix.get_ko_payoff(
            spot=105.0,
            observation_idx=1,
            accumulated_coupons=0.0,
        )

        principal = get_principal(phoenix)
        expected = principal * 0.15 * 0.30 + principal * 0.01 * 0.30
        assert payoff == expected

    def test_external_accrual_factors_drive_resolved_ko_payoff(self):
        """Test external accrual factors are used by resolved Phoenix KO profiles."""
        schedule = ObservationSchedule(
            records=[
                ObservationRecord(observation_time=0.25, barrier=103.0),
                ObservationRecord(observation_time=0.50, barrier=103.0),
            ]
        )
        barrier_config = BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=schedule,
            ki_barrier=None,
        )
        payoff_config = PayoffConfig(include_principal=False)
        accrual_config = AccrualConfig(
            is_annualized=True,
            is_annualized_ko=True,
            accrual_factors=[0.10, 0.30],
        )
        phoenix = create_test_phoenix(
            barrier_config=barrier_config,
            payoff_config=payoff_config,
            accrual_config=accrual_config,
        )

        records = phoenix.resolve_ko_observations(None)

        assert records[0].payoff == get_principal(phoenix) * 0.15 * 0.10
        assert records[1].payoff == get_principal(phoenix) * 0.15 * 0.30


class TestProperties:
    """Tests for PhoenixOption properties."""

    def test_has_memory_coupon(self):
        """Test has_memory_coupon property."""
        phoenix = create_test_phoenix()
        assert phoenix.has_memory_coupon is True

        coupon_config = create_basic_coupon_config(memory_coupon=False)
        phoenix = create_test_phoenix(coupon_config=coupon_config)
        assert phoenix.has_memory_coupon is False

    def test_has_ki_barrier(self):
        """Test has_ki_barrier property."""
        phoenix = create_test_phoenix()
        assert phoenix.has_ki_barrier is True

    def test_num_ko_observations(self):
        """Test num_ko_observations property."""
        phoenix = create_test_phoenix()
        assert phoenix.num_ko_observations == 4  # [0.25, 0.5, 0.75, 1.0]

    def test_ko_direction_standard(self):
        """Test KO direction for standard phoenix."""
        phoenix = create_test_phoenix(is_reverse=False)
        assert phoenix.get_ko_direction() == BarrierType.UP_OUT

    def test_ko_direction_reverse(self):
        """Test KO direction for reverse phoenix."""
        phoenix = create_test_phoenix(is_reverse=True)
        assert phoenix.get_ko_direction() == BarrierType.DOWN_OUT

    def test_ki_direction_standard(self):
        """Test KI direction for standard phoenix."""
        phoenix = create_test_phoenix(is_reverse=False)
        assert phoenix.get_ki_direction() == BarrierType.DOWN_IN

    def test_ki_direction_reverse(self):
        """Test KI direction for reverse phoenix."""
        phoenix = create_test_phoenix(is_reverse=True)
        assert phoenix.get_ki_direction() == BarrierType.UP_IN


class TestIntrinsicValue:
    """Tests for intrinsic value calculation."""

    def test_intrinsic_value_standard_itm(self):
        """Test intrinsic value for ITM standard phoenix (PUT)."""
        phoenix = create_test_phoenix(is_reverse=False)
        # Strike 100, spot 80 -> PUT ITM
        iv = phoenix.intrinsic_value(spot=80.0)
        assert iv == 20.0  # max(100 - 80, 0)

    def test_intrinsic_value_standard_otm(self):
        """Test intrinsic value for OTM standard phoenix (PUT)."""
        phoenix = create_test_phoenix(is_reverse=False)
        # Strike 100, spot 120 -> PUT OTM
        iv = phoenix.intrinsic_value(spot=120.0)
        assert iv == 0.0  # max(100 - 120, 0)

    def test_intrinsic_value_reverse_itm(self):
        """Test intrinsic value for ITM reverse phoenix (CALL)."""
        phoenix = create_test_phoenix(is_reverse=True)
        # Strike 100, spot 120 -> CALL ITM
        iv = phoenix.intrinsic_value(spot=120.0)
        assert iv == 20.0 * phoenix.contract_multiplier

    def test_intrinsic_value_reverse_otm(self):
        """Test intrinsic value for OTM reverse phoenix (CALL)."""
        phoenix = create_test_phoenix(is_reverse=True)
        # Strike 100, spot 80 -> CALL OTM
        iv = phoenix.intrinsic_value(spot=80.0)
        assert iv == 0.0  # max(80 - 100, 0)


class TestValidation:
    """Tests for parameter validation."""

    def test_validation_negative_initial_price(self):
        """Test validation error for negative initial price."""
        # BaseEquityOption catches this during contract multiplier validation.
        # but initial_price is invalid/missing
        with pytest.raises(ValidationError, match="Cannot derive quantity|Initial price"):
            create_test_phoenix(initial_price=-100.0)

    def test_validation_zero_strike(self):
        """Test validation error for zero strike."""
        with pytest.raises(ValidationError, match="Strike"):
            create_test_phoenix(strike=0.0)

    def test_validation_negative_contract_multiplier(self):
        """Test validation error for negative contract multiplier."""
        with pytest.raises(ValidationError, match="Contract multiplier"):
            create_test_phoenix(contract_multiplier=-10_000.0)

    def test_validation_zero_maturity(self):
        """Test validation error for zero maturity."""
        with pytest.raises(ValidationError, match="Maturity"):
            create_test_phoenix(maturity=0.0)

    def test_accrual_factors_length_mismatch_raises(self):
        """Test validation error when accrual factors do not match observations."""
        with pytest.raises(ValidationError, match="accrual_factors length"):
            create_test_phoenix(
                accrual_config=AccrualConfig(accrual_factors=[0.25])
            )

    def test_accrual_factors_invalid_values_raise(self):
        """Test validation error for invalid external accrual factors."""
        with pytest.raises(ValueError, match="accrual_factors\\[0\\]"):
            AccrualConfig(accrual_factors=[-0.25])

        with pytest.raises(ValueError, match="accrual_factors\\[0\\]"):
            AccrualConfig(accrual_factors=["bad"])


class TestHelperFunctions:
    """Tests for phoenix helper functions."""

    def test_create_standard_phoenix(self):
        """Test create_standard_phoenix helper."""
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )

        assert phoenix.initial_price == 100.0
        assert phoenix.strike == 100.0
        assert phoenix.maturity == 1.0
        assert get_principal(phoenix) == 100.0  # Default
        assert phoenix.barrier_config.ko_barrier == 103.0  # 103% default
        assert phoenix.coupon_config.coupon_barrier == 85.0  # 85% default
        assert phoenix.coupon_config.memory_coupon is True  # Default
        assert phoenix.is_reverse is False

    def test_create_stepdown_phoenix(self):
        """Test create_stepdown_phoenix helper."""
        phoenix = create_stepdown_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            num_observations=4,
            ko_stepdown_rate=0.01,
            coupon_stepdown_rate=0.01,
        )

        # Check KO barriers are decreasing
        ko_barriers = phoenix.barrier_config.ko_barrier
        assert isinstance(ko_barriers, list)
        assert len(ko_barriers) == 4
        assert ko_barriers[0] > ko_barriers[1] > ko_barriers[2] > ko_barriers[3]

        # Check coupon barriers are decreasing
        coupon_barriers = phoenix.coupon_config.coupon_barrier
        assert isinstance(coupon_barriers, list)
        assert len(coupon_barriers) == 4

    def test_create_reverse_phoenix(self):
        """Test create_reverse_phoenix helper."""
        phoenix = create_reverse_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )

        assert phoenix.is_reverse is True
        assert phoenix.option_type == OptionType.CALL
        assert phoenix.barrier_config.ko_barrier == 97.0  # 97% default for reverse
        assert phoenix.barrier_config.ki_barrier == 125.0  # 125% default for reverse
        # Use approximate comparison for floating point
        assert abs(phoenix.coupon_config.coupon_barrier - 115.0) < 0.01  # 115% for reverse

    def test_create_memory_phoenix(self):
        """Test create_memory_phoenix helper."""
        phoenix = create_memory_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )

        assert phoenix.coupon_config.memory_coupon is True
        assert phoenix.has_memory_coupon is True

    def test_create_non_memory_phoenix(self):
        """Test create_non_memory_phoenix helper."""
        phoenix = create_non_memory_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )

        assert phoenix.coupon_config.memory_coupon is False
        assert phoenix.has_memory_coupon is False

    def test_helper_custom_day_count(self):
        """Test helper with custom day count convention."""
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            day_count_convention=DayCountConvention.THIRTY_360_US,
        )

        assert phoenix.coupon_config.day_count_convention == DayCountConvention.THIRTY_360_US

    def test_helper_custom_coupon_pay_type(self):
        """Test helper with custom coupon pay type."""
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            coupon_pay_type=CouponPayType.EXPIRY,
        )

        assert phoenix.coupon_config.coupon_pay_type == CouponPayType.EXPIRY

    def test_helper_validation_error(self):
        """Test helper raises validation error for invalid params."""
        with pytest.raises(ValidationError, match="initial_price"):
            create_standard_phoenix(
                initial_price=-100.0,
                strike=100.0,
                maturity=1.0,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
