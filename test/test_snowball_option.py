"""
Comprehensive unit tests for SnowballOption.

Tests cover:
- Basic product creation (standard and reverse)
- Option type and exercise type inheritance
- Barrier configuration and triggering
- Payoff calculations (V0 and V1 states)
- Validation errors
- Date-based and maturity-based specification
- Base class method inheritance
- Edge cases and boundary conditions
"""

import sys
from pathlib import Path
import pytest
from datetime import datetime
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.product.option.snowball_option import SnowballOption
from asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
    AccrualConfig,
)
from util.enum import (
    ObservationType,
    OptionType,
    ExerciseType,
    CouponPayType,
    ProtectionType,
    TenorEnd,
    BarrierType,
)
from util.calendar import DayCountConvention
from util.exceptions import ValidationError


# =============================================================================
# Fixtures - Common test configurations
# =============================================================================


def create_basic_barrier_config(
    ko_barrier: float = 1.03,
    ko_rate: float = 0.15,
    ki_barrier: float = 0.75,
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


def create_standard_snowball(
    initial_price: float = 100.0,
    strike: float = 100.0,
    notional: float = 1_000_000.0,
    maturity: float = 1.0,
    barrier_config: BarrierConfig = None,
    payoff_config: PayoffConfig = None,
) -> SnowballOption:
    """Create a standard snowball option for testing."""
    if barrier_config is None:
        barrier_config = create_basic_barrier_config()

    return SnowballOption(
        initial_price=initial_price,
        strike=strike,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        notional=notional,
        maturity=maturity,
        is_reverse=False,
    )


def create_reverse_snowball(
    initial_price: float = 100.0,
    strike: float = 100.0,
    notional: float = 1_000_000.0,
    maturity: float = 1.0,
    barrier_config: BarrierConfig = None,
    payoff_config: PayoffConfig = None,
) -> SnowballOption:
    """Create a reverse snowball option for testing."""
    if barrier_config is None:
        barrier_config = create_basic_barrier_config()

    return SnowballOption(
        initial_price=initial_price,
        strike=strike,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        notional=notional,
        maturity=maturity,
        is_reverse=True,
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestSnowballCreation:
    """Tests for basic snowball option creation."""

    def test_standard_snowball_creation(self):
        """Test basic standard snowball creation."""
        snowball = create_standard_snowball()

        assert snowball.initial_price == 100.0
        assert snowball.strike == 100.0
        assert snowball.notional == 1_000_000.0
        assert snowball.maturity == 1.0
        assert snowball.is_reverse is False
        assert snowball.is_standard is True

    def test_reverse_snowball_creation(self):
        """Test basic reverse snowball creation."""
        snowball = create_reverse_snowball()

        assert snowball.initial_price == 100.0
        assert snowball.strike == 100.0
        assert snowball.notional == 1_000_000.0
        assert snowball.maturity == 1.0
        assert snowball.is_reverse is True
        assert snowball.is_standard is False

    def test_date_based_creation(self):
        """Test snowball creation with date-based specification."""
        barrier_config = create_basic_barrier_config()

        snowball = SnowballOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=barrier_config,
            notional=1_000_000.0,
            initial_date=datetime(2024, 1, 1),
            exercise_date=datetime(2025, 1, 1),
            settlement_date=datetime(2025, 1, 3),
        )

        assert snowball.initial_date == datetime(2024, 1, 1)
        assert snowball.exercise_date == datetime(2025, 1, 1)
        assert snowball.settlement_date == datetime(2025, 1, 3)

    def test_tenor_specification(self):
        """Test snowball creation with explicit tenor."""
        barrier_config = create_basic_barrier_config()

        snowball = SnowballOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=barrier_config,
            notional=1_000_000.0,
            maturity=0.5,
            tenor=1.0,
        )

        assert snowball.maturity == 0.5
        assert snowball.tenor == 1.0
        assert snowball.get_maturity() == 0.5
        assert snowball.get_tenor() == 1.0


class TestOptionTypeInheritance:
    """Tests for option type and exercise type inheritance from BaseEquityOption."""

    def test_standard_snowball_option_type(self):
        """Test that standard snowball has PUT option type."""
        snowball = create_standard_snowball()

        assert snowball.option_type == OptionType.PUT
        assert snowball.is_put() is True
        assert snowball.is_call() is False

    def test_reverse_snowball_option_type(self):
        """Test that reverse snowball has CALL option type."""
        snowball = create_reverse_snowball()

        assert snowball.option_type == OptionType.CALL
        assert snowball.is_call() is True
        assert snowball.is_put() is False

    def test_exercise_type_is_european(self):
        """Test that snowballs are European-style."""
        standard = create_standard_snowball()
        reverse = create_reverse_snowball()

        assert standard.exercise_type == ExerciseType.EUROPEAN
        assert standard.is_european() is True
        assert standard.is_american() is False

        assert reverse.exercise_type == ExerciseType.EUROPEAN
        assert reverse.is_european() is True


class TestBarrierConfiguration:
    """Tests for barrier configuration and triggering."""

    def test_ko_barrier_triggering_standard(self):
        """Test KO barrier trigger for standard snowball (up barrier).
        
        Note: Barriers are specified as ratios. is_ko_triggered compares spot to barrier directly.
        So if barrier=1.03, spot must be >= 1.03 to trigger.
        """
        snowball = create_standard_snowball()

        # KO barrier at 1.03 (ratio format)
        assert snowball.is_ko_triggered(1.03) is True  # At barrier
        assert snowball.is_ko_triggered(1.05) is True  # Above barrier
        assert snowball.is_ko_triggered(1.02) is False  # Below barrier
        assert snowball.is_ko_triggered(1.00) is False  # At initial

    def test_ki_barrier_triggering_standard(self):
        """Test KI barrier trigger for standard snowball (down barrier).
        
        Note: Barriers are specified as ratios. is_ki_triggered compares spot to barrier directly.
        """
        snowball = create_standard_snowball()

        # KI barrier at 0.75 (ratio format)
        assert snowball.is_ki_triggered(0.75) is True  # At barrier
        assert snowball.is_ki_triggered(0.70) is True  # Below barrier
        assert snowball.is_ki_triggered(0.80) is False  # Above barrier
        assert snowball.is_ki_triggered(1.00) is False  # At initial

    def test_has_ki_barrier(self):
        """Test has_ki_barrier property."""
        snowball_with_ki = create_standard_snowball()
        assert snowball_with_ki.has_ki_barrier is True

        # Create snowball without KI barrier
        barrier_config = BarrierConfig(
            ko_barrier=1.03,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=None,
        )
        snowball_without_ki = SnowballOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=barrier_config,
            notional=1_000_000.0,
            maturity=1.0,
        )
        assert snowball_without_ki.has_ki_barrier is False

    def test_time_varying_barriers(self):
        """Test time-varying barrier levels."""
        barrier_config = BarrierConfig(
            ko_barrier=[1.03, 1.02, 1.01, 1.00],  # Decreasing barriers
            ko_rate=[0.10, 0.12, 0.14, 0.16],  # Increasing rates
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=[0.80, 0.78, 0.76, 0.75],  # Decreasing barriers
            ki_observation_type=ObservationType.DISCRETE,  # Need discrete for KI arrays
            ki_observation_dates=[0.25, 0.5, 0.75, 1.0],  # Add observation dates
        )

        snowball = SnowballOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=barrier_config,
            notional=1_000_000.0,
            maturity=1.0,
        )

        # Check barrier at different indices
        assert snowball.get_ko_barrier_at(0) == 1.03
        assert snowball.get_ko_barrier_at(3) == 1.00
        assert snowball.get_ko_rate_at(0) == 0.10
        assert snowball.get_ko_rate_at(3) == 0.16
        assert snowball.get_ki_barrier_at(0) == 0.80
        assert snowball.get_ki_barrier_at(3) == 0.75

    def test_get_ko_direction_standard(self):
        """Test KO barrier direction for standard snowball."""
        snowball = create_standard_snowball()
        assert snowball.get_ko_direction() == BarrierType.UP_OUT

    def test_get_ko_direction_reverse(self):
        """Test KO barrier direction for reverse snowball."""
        snowball = create_reverse_snowball()
        assert snowball.get_ko_direction() == BarrierType.DOWN_OUT

    def test_get_ki_direction_standard(self):
        """Test KI barrier direction for standard snowball."""
        snowball = create_standard_snowball()
        assert snowball.get_ki_direction() == BarrierType.DOWN_IN

    def test_get_ki_direction_reverse(self):
        """Test KI barrier direction for reverse snowball."""
        snowball = create_reverse_snowball()
        assert snowball.get_ki_direction() == BarrierType.UP_IN

    def test_observation_counts(self):
        """Test observation count properties."""
        snowball = create_standard_snowball()

        assert snowball.num_ko_observations == 4
        # KI is continuous, so observation count depends on schedule
        assert snowball.num_ki_observations == 0  # No discrete KI schedule


class TestPayoffCalculations:
    """Tests for payoff calculations."""

    def test_get_payoff_dispatches_to_v0(self):
        """Test that get_payoff returns V0 payoff when not knocked in."""
        payoff_config = PayoffConfig(
            rebate_rate=0.10,
            include_principal=True,
        )
        accrual_config = AccrualConfig(
            is_annualized=False,
            is_annualized_rebate=False,
        )
        snowball = create_standard_snowball(payoff_config=payoff_config)
        snowball.accrual_config = accrual_config

        # With knocked_in=False, should get V0 payoff
        payoff = snowball.get_payoff(spot=100.0, knocked_in=False)
        v0_payoff = snowball.get_maturity_payoff_v0(spot=100.0)

        assert payoff == v0_payoff, f"get_payoff should equal V0 when not knocked in"

    def test_get_payoff_dispatches_to_v1(self):
        """Test that get_payoff returns V1 payoff when knocked in."""
        payoff_config = PayoffConfig(
            participation_rate=1.0,
            include_principal=True,
            protection_type=ProtectionType.NONE,
        )
        accrual_config = AccrualConfig(
            is_annualized=False,
            is_annualized_ki=False,
        )
        snowball = create_standard_snowball(payoff_config=payoff_config)
        snowball.accrual_config = accrual_config

        # With knocked_in=True, should get V1 payoff
        payoff = snowball.get_payoff(spot=90.0, knocked_in=True)
        v1_payoff = snowball.get_maturity_payoff_v1(spot=90.0)

        assert payoff == v1_payoff, f"get_payoff should equal V1 when knocked in"

    def test_get_payoff_default_is_v0(self):
        """Test that get_payoff defaults to V0 (not knocked in)."""
        snowball = create_standard_snowball()

        # Without specifying knocked_in, should get V0
        payoff_default = snowball.get_payoff(spot=100.0)
        payoff_v0 = snowball.get_payoff(spot=100.0, knocked_in=False)

        assert payoff_default == payoff_v0, "Default should be V0 payoff"

    def test_get_payoff_negative_spot_raises(self):
        """Test that get_payoff raises ValidationError for negative spot."""
        snowball = create_standard_snowball()

        with pytest.raises(ValidationError, match="Spot price must be non-negative"):
            snowball.get_payoff(spot=-10.0)

    def test_get_payoff_v0_vs_v1_different(self):
        """Test that V0 and V1 payoffs are different at same spot."""
        payoff_config = PayoffConfig(
            rebate_rate=0.10,
            participation_rate=1.0,
            include_principal=True,
        )
        accrual_config = AccrualConfig(
            is_annualized=False,
            is_annualized_ki=False,
            is_annualized_rebate=False,
        )
        snowball = create_standard_snowball(payoff_config=payoff_config)
        snowball.accrual_config = accrual_config

        # At spot = 90 (below strike), V0 and V1 should differ
        payoff_v0 = snowball.get_payoff(spot=90.0, knocked_in=False)
        payoff_v1 = snowball.get_payoff(spot=90.0, knocked_in=True)

        assert payoff_v0 != payoff_v1, f"V0={payoff_v0}, V1={payoff_v1} should differ"

    def test_v1_payoff_standard_snowball(self):
        """Test V1 (KI triggered) payoff for standard snowball."""
        payoff_config = PayoffConfig(
            participation_rate=1.0,
            include_principal=True,
            protection_type=ProtectionType.NONE,
        )
        accrual_config = AccrualConfig(
            is_annualized=False,
            is_annualized_ki=False,
        )
        snowball = create_standard_snowball(
            payoff_config=payoff_config,
        )
        snowball.accrual_config = accrual_config

        # V1 payoff when spot = 90 (below strike of 100)
        # For standard: participation × (Spot - Strike) = 1.0 × (90 - 100) = -10
        v1_payoff = snowball.get_maturity_payoff_v1(spot=90.0)

        # Principal + participation * (spot - strike)
        expected = 1_000_000.0 + 1.0 * (90.0 - 100.0)  # = 999,990
        assert abs(v1_payoff - expected) < 0.01, f"Expected {expected}, got {v1_payoff}"

    def test_v1_payoff_with_protection(self):
        """Test V1 payoff with partial protection."""
        payoff_config = PayoffConfig(
            participation_rate=1.0,
            include_principal=True,
            protection_type=ProtectionType.PARTIAL,
            protection_rate=0.50,  # 50% protection
        )
        accrual_config = AccrualConfig(
            is_annualized=False,
            is_annualized_ki=False,
        )
        snowball = create_standard_snowball(
            payoff_config=payoff_config,
        )
        snowball.accrual_config = accrual_config

        # V1 payoff with 50% protection
        # Downside floor = -protection_rate * notional = -500,000
        # When spot = 50 (deep below strike)
        # Unprotected downside = 1.0 * (50 - 100) = -50
        # Protected downside is max(-50, -500,000) = -50 (not floored since small)
        v1_payoff = snowball.get_maturity_payoff_v1(spot=50.0)

        expected = 1_000_000.0 + max(1.0 * (50.0 - 100.0), -500_000.0)
        assert abs(v1_payoff - expected) < 0.01, f"Expected {expected}, got {v1_payoff}"

    def test_v1_payoff_spot_above_strike(self):
        """Test V1 payoff when spot is above strike (Standard: Short Put is OTM -> No Loss)."""
        payoff_config = PayoffConfig(
            participation_rate=1.0,
            include_principal=True,
            protection_type=ProtectionType.NONE,
        )
        accrual_config = AccrualConfig(
            is_annualized=False,
            is_annualized_ki=False,
        )
        snowball = create_standard_snowball(payoff_config=payoff_config)
        snowball.accrual_config = accrual_config

        # V1 payoff when spot = 110 (above strike of 100)
        # Standard: Short Put is OTM. Downside = min(110 - 100, 0) = 0.
        v1_payoff = snowball.get_maturity_payoff_v1(spot=110.0)

        expected = 1_000_000.0 + 0.0  # = 1,000,000
        assert abs(v1_payoff - expected) < 0.01, f"Expected {expected}, got {v1_payoff}"

    def test_v1_payoff_without_principal(self):
        """Test V1 payoff without principal inclusion."""
        payoff_config = PayoffConfig(
            participation_rate=1.0,
            include_principal=False,  # No principal
            protection_type=ProtectionType.NONE,
        )
        accrual_config = AccrualConfig(
            is_annualized=False,
            is_annualized_ki=False,
        )
        snowball = create_standard_snowball(payoff_config=payoff_config)
        snowball.accrual_config = accrual_config

        # V1 payoff when spot = 90
        v1_payoff = snowball.get_maturity_payoff_v1(spot=90.0)

        # No principal, just participation * (spot - strike)
        expected = 0.0 + 1.0 * (90.0 - 100.0)  # = -10
        assert abs(v1_payoff - expected) < 0.01, f"Expected {expected}, got {v1_payoff}"

    def test_v1_payoff_with_participation_rate(self):
        """Test V1 payoff with different participation rate."""
        payoff_config = PayoffConfig(
            participation_rate=0.5,  # 50% participation
            include_principal=True,
            protection_type=ProtectionType.NONE,
        )
        accrual_config = AccrualConfig(
            is_annualized=False,
            is_annualized_ki=False,
        )
        snowball = create_standard_snowball(payoff_config=payoff_config)
        snowball.accrual_config = accrual_config

        # V1 payoff when spot = 80
        # participation × (Spot - Strike) = 0.5 × (80 - 100) = -10
        v1_payoff = snowball.get_maturity_payoff_v1(spot=80.0)

        expected = 1_000_000.0 + 0.5 * (80.0 - 100.0)  # = 999,990
        assert abs(v1_payoff - expected) < 0.01, f"Expected {expected}, got {v1_payoff}"

    def test_v1_payoff_reverse_snowball(self):
        """Test V1 payoff for reverse snowball (Short Call on KI)."""
        payoff_config = PayoffConfig(
            participation_rate=1.0,
            include_principal=True,
            protection_type=ProtectionType.NONE,
        )
        accrual_config = AccrualConfig(
            is_annualized=False,
            is_annualized_ki=False,
        )
        snowball = create_reverse_snowball(
            payoff_config=payoff_config,
            strike=100.0,
        )
        snowball.accrual_config = accrual_config

        # Case 1: Spot > Strike (Loss for Short Call)
        # Spot = 110.0, Strike = 100.0
        # Payoff = Principal + 1.0 * (Strike - Spot) = 1,000,000 + (100 - 110) = 999,990
        v1_payoff_loss = snowball.get_maturity_payoff_v1(spot=110.0)
        expected_loss = 1_000_000.0 + (100.0 - 110.0)
        assert abs(v1_payoff_loss - expected_loss) < 0.01, f"Loss case: Expected {expected_loss}, got {v1_payoff_loss}"

        # Case 2: Spot < Strike (Gain/No Loss for Short Call)
        # Spot = 90.0, Strike = 100.0
        # Reverse: Short Call is OTM. Downside = min(100 - 90, 0) = 0.
        v1_payoff_gain = snowball.get_maturity_payoff_v1(spot=90.0)
        expected_gain = 1_000_000.0 + 0.0
        assert abs(v1_payoff_gain - expected_gain) < 0.01, f"Gain case: Expected {expected_gain}, got {v1_payoff_gain}"


class TestIntrinsicValue:
    """Tests for intrinsic value calculations."""

    def test_standard_snowball_intrinsic_value(self):
        """Test intrinsic value for standard snowball (PUT)."""
        snowball = create_standard_snowball()

        # ITM: spot < strike
        assert snowball.intrinsic_value(90.0) == 10.0
        # ATM: spot == strike
        assert snowball.intrinsic_value(100.0) == 0.0
        # OTM: spot > strike
        assert snowball.intrinsic_value(110.0) == 0.0

    def test_reverse_snowball_intrinsic_value(self):
        """Test intrinsic value for reverse snowball (CALL)."""
        snowball = create_reverse_snowball()

        # OTM: spot < strike
        assert snowball.intrinsic_value(90.0) == 0.0
        # ATM: spot == strike
        assert snowball.intrinsic_value(100.0) == 0.0
        # ITM: spot > strike
        assert snowball.intrinsic_value(110.0) == 10.0

    def test_intrinsic_value_negative_spot_raises(self):
        """Test that negative spot raises ValidationError."""
        snowball = create_standard_snowball()

        with pytest.raises(ValidationError):
            snowball.intrinsic_value(-10.0)


class TestBaseClassMethods:
    """Tests for inherited base class methods."""

    def test_get_maturity(self):
        """Test get_maturity method."""
        snowball = create_standard_snowball(maturity=0.5)
        assert snowball.get_maturity() == 0.5

    def test_get_tenor(self):
        """Test get_tenor method."""
        barrier_config = create_basic_barrier_config()
        snowball = SnowballOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=barrier_config,
            notional=1_000_000.0,
            maturity=0.5,
            tenor=1.0,
        )
        assert snowball.get_tenor() == 1.0

    def test_get_tenor_fallback_to_maturity(self):
        """Test get_tenor falls back to maturity when tenor not set."""
        snowball = create_standard_snowball(maturity=0.75)
        assert snowball.get_tenor() == 0.75

    def test_get_notional(self):
        """Test get_notional method."""
        snowball = create_standard_snowball(notional=2_000_000.0)
        assert snowball.get_notional() == 2_000_000.0

    def test_moneyness(self):
        """Test moneyness calculation."""
        snowball = create_standard_snowball()

        assert snowball.moneyness(100.0) == 1.0  # ATM
        assert snowball.moneyness(110.0) == 1.1  # OTM for put
        assert snowball.moneyness(90.0) == 0.9  # ITM for put

    def test_is_in_the_money_standard(self):
        """Test ITM check for standard snowball (PUT)."""
        snowball = create_standard_snowball()

        # For PUT: ITM when spot < strike
        assert snowball.is_in_the_money(90.0) is True
        assert snowball.is_in_the_money(110.0) is False

    def test_is_in_the_money_reverse(self):
        """Test ITM check for reverse snowball (CALL)."""
        snowball = create_reverse_snowball()

        # For CALL: ITM when spot > strike
        assert snowball.is_in_the_money(110.0) is True
        assert snowball.is_in_the_money(90.0) is False


class TestValidationErrors:
    """Tests for validation error handling."""

    def test_negative_initial_price_raises(self):
        """Test that negative initial price raises ValidationError."""
        barrier_config = create_basic_barrier_config()

        with pytest.raises(ValidationError, match="Initial price must be positive"):
            SnowballOption(
                initial_price=-100.0,
                strike=100.0,
                barrier_config=barrier_config,
                notional=1_000_000.0,
                maturity=1.0,
            )

    def test_negative_strike_raises(self):
        """Test that negative strike raises ValidationError."""
        barrier_config = create_basic_barrier_config()

        with pytest.raises(ValidationError, match="Strike must be positive"):
            SnowballOption(
                initial_price=100.0,
                strike=-100.0,
                barrier_config=barrier_config,
                notional=1_000_000.0,
                maturity=1.0,
            )

    def test_negative_notional_raises(self):
        """Test that negative notional raises ValidationError."""
        barrier_config = create_basic_barrier_config()

        with pytest.raises(ValidationError, match="Notional must be positive"):
            SnowballOption(
                initial_price=100.0,
                strike=100.0,
                barrier_config=barrier_config,
                notional=-1_000_000.0,
                maturity=1.0,
            )

    def test_negative_maturity_raises(self):
        """Test that negative maturity raises ValidationError."""
        barrier_config = create_basic_barrier_config()

        with pytest.raises(ValidationError, match="Maturity must be positive"):
            SnowballOption(
                initial_price=100.0,
                strike=100.0,
                barrier_config=barrier_config,
                notional=1_000_000.0,
                maturity=-1.0,
            )

    def test_missing_maturity_and_exercise_date_passes(self):
        """Test that exercise_date alone is sufficient (no maturity needed)."""
        barrier_config = create_basic_barrier_config()

        # Should work with exercise_date alone
        snowball = SnowballOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=barrier_config,
            notional=1_000_000.0,
            exercise_date=datetime(2025, 1, 1),
        )
        assert snowball.exercise_date == datetime(2025, 1, 1)

    def test_discrete_ko_without_dates_raises(self):
        """Test that discrete KO without observation dates raises ValidationError."""
        barrier_config = BarrierConfig(
            ko_barrier=1.03,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=None,  # Missing dates
        )

        with pytest.raises(ValidationError, match="KO observation dates"):
            SnowballOption(
                initial_price=100.0,
                strike=100.0,
                barrier_config=barrier_config,
                notional=1_000_000.0,
                maturity=1.0,
            )


class TestConfigurationObjects:
    """Tests for configuration object handling."""

    def test_default_payoff_config(self):
        """Test that default PayoffConfig is applied."""
        snowball = create_standard_snowball()

        assert snowball.payoff_config is not None
        assert snowball.payoff_config.include_principal is True
        assert snowball.payoff_config.participation_rate == 1.0

    def test_default_accrual_config(self):
        """Test that default AccrualConfig is applied."""
        snowball = create_standard_snowball()

        assert snowball.accrual_config is not None
        assert snowball.accrual_config.coupon_pay_type == CouponPayType.INSTANT

    def test_custom_payoff_config(self):
        """Test custom PayoffConfig."""
        payoff_config = PayoffConfig(
            rebate_rate=0.15,
            participation_rate=0.8,
            include_principal=False,
            protection_type=ProtectionType.FULL,
        )

        snowball = create_standard_snowball(payoff_config=payoff_config)

        assert snowball.payoff_config.rebate_rate == 0.15
        assert snowball.payoff_config.participation_rate == 0.8
        assert snowball.payoff_config.include_principal is False
        assert snowball.payoff_config.protection_type == ProtectionType.FULL


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_at_the_money_barrier(self):
        """Test behavior when spot is exactly at barrier (ratio format)."""
        snowball = create_standard_snowball()

        # Exactly at KO barrier (1.03 ratio)
        assert snowball.is_ko_triggered(1.03) is True

        # Exactly at KI barrier (0.75 ratio)
        assert snowball.is_ki_triggered(0.75) is True

    def test_zero_strike(self):
        """Test that zero strike raises ValidationError."""
        barrier_config = create_basic_barrier_config()

        with pytest.raises(ValidationError):
            SnowballOption(
                initial_price=100.0,
                strike=0.0,
                barrier_config=barrier_config,
                notional=1_000_000.0,
                maturity=1.0,
            )

    def test_very_small_maturity(self):
        """Test with very small maturity."""
        snowball = create_standard_snowball(maturity=0.01)  # About 4 days

        assert snowball.get_maturity() == 0.01

    def test_long_maturity(self):
        """Test with long maturity."""
        snowball = create_standard_snowball(maturity=10.0)  # 10 years

        assert snowball.get_maturity() == 10.0

    def test_high_notional(self):
        """Test with high notional value."""
        snowball = create_standard_snowball(notional=1_000_000_000.0)  # 1 billion

        assert snowball.notional == 1_000_000_000.0
        assert snowball.get_notional() == 1_000_000_000.0


class TestReprAndStr:
    """Tests for string representation."""

    def test_repr_standard_snowball(self):
        """Test __repr__ for standard snowball."""
        snowball = create_standard_snowball()

        repr_str = repr(snowball)
        assert "SnowballOption" in repr_str
        # Check that it contains key information
        assert "100.0" in repr_str or "1000000" in repr_str

    def test_repr_reverse_snowball(self):
        """Test __repr__ for reverse snowball."""
        snowball = create_reverse_snowball()

        repr_str = repr(snowball)
        assert "SnowballOption" in repr_str
        # Check that it contains key information
        assert "100.0" in repr_str or "1000000" in repr_str


# =============================================================================
# Test Runner
# =============================================================================


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "=" * 70)
    print("Running Snowball Option Comprehensive Tests")
    print("=" * 70 + "\n")

    test_classes = [
        TestSnowballCreation,
        TestOptionTypeInheritance,
        TestBarrierConfiguration,
        TestPayoffCalculations,
        TestIntrinsicValue,
        TestBaseClassMethods,
        TestValidationErrors,
        TestConfigurationObjects,
        TestEdgeCases,
        TestReprAndStr,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 50)

        test_instance = test_class()

        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                method = getattr(test_instance, method_name)
                try:
                    method()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    errors.append((test_class.__name__, method_name, str(e)))
                    failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if errors:
        print("\nFailed Tests:")
        for class_name, method, error in errors:
            print(f"  - {class_name}.{method}: {error}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
