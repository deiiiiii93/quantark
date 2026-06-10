"""
Unit tests for Range Accrual option product.

Tests cover:
- Configuration validation (barriers, rates, enums)
- Product creation (standard, reverse, stepdown)
- Payoff calculation (100%, 50%, 0% in-range)
- Weighted accrual calculation (Friday=3 calendar day convention)
- Historical observation handling
- Helper functions
"""

import pytest
from datetime import datetime

from quantark.asset.equity.product.option import (
    RangeAccrualOption,
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
    create_standard_range_accrual,
    create_reverse_range_accrual,
    create_stepdown_range_accrual,
    generate_range_observation_records,
    assign_calendar_day_weights,
)
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationFrequency
from quantark.util.exceptions import ValidationError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_pricing_env():
    """Create a basic pricing environment for testing."""
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.05),
        valuation_date=datetime(2025, 1, 1),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(0.20),
    )


@pytest.fixture
def basic_range_config():
    """Create a basic range accrual config for testing."""
    return RangeAccrualConfig(
        upper_barrier=110.0,
        lower_barrier=90.0,
        accrual_rate=0.05,
        is_rate_annualized=True,
        day_count_convention=DayCountConvention.ACT_365,
        accrual_pay_type=CouponPayType.EXPIRY,
    )


# =============================================================================
# RangeAccrualObservationRecord Tests
# =============================================================================


class TestRangeAccrualObservationRecord:
    """Tests for RangeAccrualObservationRecord dataclass."""

    def test_create_with_time(self):
        """Test creating record with observation_time."""
        record = RangeAccrualObservationRecord(observation_time=0.5, weight=1.0)
        assert record.observation_time == 0.5
        assert record.weight == 1.0
        assert record.observed_in_range is None
        assert not record.is_observed()

    def test_create_with_date(self):
        """Test creating record with observation_date."""
        record = RangeAccrualObservationRecord(
            observation_date=datetime(2025, 6, 15), weight=3.0
        )
        assert record.observation_date == datetime(2025, 6, 15)
        assert record.weight == 3.0

    def test_create_with_historical_data(self):
        """Test creating record with historical observation."""
        record = RangeAccrualObservationRecord(
            observation_time=-0.1, weight=1.0, observed_in_range=True
        )
        assert record.observed_in_range is True
        assert record.is_observed()

    def test_create_with_per_observation_barriers(self):
        """Test creating record with per-observation barriers."""
        record = RangeAccrualObservationRecord(
            observation_time=0.25,
            upper_barrier=115.0,
            lower_barrier=85.0,
            weight=1.0,
        )
        assert record.upper_barrier == 115.0
        assert record.lower_barrier == 85.0

    def test_validate_missing_time_and_date(self):
        """Test validation fails when neither time nor date provided."""
        record = RangeAccrualObservationRecord(weight=1.0)
        with pytest.raises(ValidationError, match="observation_time or observation_date"):
            record.validate()

    def test_validate_negative_weight(self):
        """Test validation fails for non-positive weight."""
        record = RangeAccrualObservationRecord(observation_time=0.5, weight=0.0)
        with pytest.raises(ValidationError, match="weight must be positive"):
            record.validate()

    def test_validate_barrier_ordering(self):
        """Test validation fails when lower >= upper barrier."""
        record = RangeAccrualObservationRecord(
            observation_time=0.5,
            upper_barrier=90.0,
            lower_barrier=110.0,  # Wrong order
            weight=1.0,
        )
        with pytest.raises(ValidationError, match="lower_barrier.*must be less than"):
            record.validate()

    def test_resolve_time_with_year_fraction(self, basic_pricing_env):
        """Test resolving time from observation_time."""
        record = RangeAccrualObservationRecord(observation_time=0.5)
        t = record.resolve_time(basic_pricing_env)
        assert t == 0.5

    def test_resolve_time_from_future_date(self, basic_pricing_env):
        """Test resolving time from future observation_date."""
        record = RangeAccrualObservationRecord(
            observation_date=datetime(2025, 7, 1)  # 6 months from Jan 1
        )
        t = record.resolve_time(basic_pricing_env)
        assert 0.45 < t < 0.55  # Approximately 0.5 years

    def test_resolve_time_from_past_date(self, basic_pricing_env):
        """Test resolving time from past observation_date (negative)."""
        record = RangeAccrualObservationRecord(
            observation_date=datetime(2024, 7, 1)  # 6 months before Jan 1
        )
        t = record.resolve_time(basic_pricing_env)
        assert -0.55 < t < -0.45  # Approximately -0.5 years


# =============================================================================
# RangeAccrualConfig Tests
# =============================================================================


class TestRangeAccrualConfig:
    """Tests for RangeAccrualConfig dataclass."""

    def test_create_basic_config(self):
        """Test creating basic config with scalar barriers."""
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
        )
        assert config.upper_barrier == 110.0
        assert config.lower_barrier == 90.0
        assert config.accrual_rate == 0.05
        assert config.is_reverse is False

    def test_create_config_with_list_barriers(self):
        """Test creating config with time-varying barriers."""
        config = RangeAccrualConfig(
            upper_barrier=[110.0, 108.0, 106.0],
            lower_barrier=[90.0, 92.0, 94.0],
            accrual_rate=0.05,
        )
        assert len(config.upper_barrier) == 3
        assert config.get_upper_barrier(0) == 110.0
        assert config.get_upper_barrier(2) == 106.0
        assert config.get_lower_barrier(0) == 90.0
        assert config.get_lower_barrier(2) == 94.0

    def test_create_reverse_config(self):
        """Test creating reverse config (pay outside range)."""
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_reverse=True,
        )
        assert config.is_reverse is True

    def test_validate_negative_barrier(self):
        """Test validation fails for non-positive barrier."""
        with pytest.raises(ValidationError, match="must be positive"):
            RangeAccrualConfig(
                upper_barrier=-110.0,
                lower_barrier=90.0,
                accrual_rate=0.05,
            )

    def test_validate_barrier_ordering(self):
        """Test validation fails when lower >= upper barrier."""
        with pytest.raises(ValidationError, match="must be less than"):
            RangeAccrualConfig(
                upper_barrier=90.0,
                lower_barrier=110.0,  # Wrong order
                accrual_rate=0.05,
            )

    def test_validate_list_barrier_ordering(self):
        """Test validation fails when any list element has wrong ordering."""
        with pytest.raises(ValidationError, match="must be less than"):
            RangeAccrualConfig(
                upper_barrier=[110.0, 108.0, 100.0],  # Third element
                lower_barrier=[90.0, 92.0, 100.0],  # crosses
                accrual_rate=0.05,
            )

    def test_validate_list_length_mismatch(self):
        """Test validation fails when barrier lists have different lengths."""
        with pytest.raises(ValidationError, match="same length"):
            RangeAccrualConfig(
                upper_barrier=[110.0, 108.0],
                lower_barrier=[90.0, 92.0, 94.0],  # Different length
                accrual_rate=0.05,
            )

    def test_validate_negative_accrual_rate(self):
        """Test validation fails for negative accrual rate."""
        with pytest.raises(ValidationError, match="non-negative"):
            RangeAccrualConfig(
                upper_barrier=110.0,
                lower_barrier=90.0,
                accrual_rate=-0.05,
            )


# =============================================================================
# RangeAccrualOption Tests
# =============================================================================


class TestRangeAccrualOption:
    """Tests for RangeAccrualOption product class."""

    def test_create_with_num_observations(self, basic_range_config):
        """Test creating option with num_observations."""
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            num_observations=12,
        )
        assert option.initial_price == 100.0
        assert option.maturity == 1.0
        assert option.num_observations == 12

    def test_create_with_observation_times(self, basic_range_config):
        """Test creating option with observation_times."""
        obs_times = [0.25, 0.5, 0.75, 1.0]
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            observation_times=obs_times,
        )
        assert option.observation_times == obs_times

    def test_create_with_observation_records(self, basic_range_config):
        """Test creating option with observation_records."""
        records = [
            RangeAccrualObservationRecord(observation_time=0.25, weight=1.0),
            RangeAccrualObservationRecord(observation_time=0.5, weight=1.0),
            RangeAccrualObservationRecord(observation_time=0.75, weight=3.0),  # Friday
            RangeAccrualObservationRecord(observation_time=1.0, weight=1.0),
        ]
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            observation_records=records,
        )
        assert len(option.observation_records) == 4
        assert option.get_total_weights() == 6.0

    def test_validate_missing_observations(self, basic_range_config):
        """Test validation fails without any observation specification."""
        with pytest.raises(ValidationError, match="observation_records, observation_times, or num_observations"):
            RangeAccrualOption(
                initial_price=100.0,
                range_config=basic_range_config,
                maturity=1.0,
            )

    def test_validate_conflicting_observations(self, basic_range_config):
        """Test validation fails with both records and times."""
        with pytest.raises(ValidationError, match="Cannot provide both"):
            RangeAccrualOption(
                initial_price=100.0,
                range_config=basic_range_config,
                maturity=1.0,
                observation_times=[0.5, 1.0],
                observation_records=[
                    RangeAccrualObservationRecord(observation_time=0.5),
                ],
            )

    def test_is_in_range_inside(self, basic_range_config):
        """Test is_in_range returns True when spot is inside range."""
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            num_observations=12,
        )
        assert option.is_in_range(100.0, 0) is True  # Exactly in middle
        assert option.is_in_range(90.0, 0) is True  # At lower boundary
        assert option.is_in_range(110.0, 0) is True  # At upper boundary

    def test_is_in_range_outside(self, basic_range_config):
        """Test is_in_range returns False when spot is outside range."""
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            num_observations=12,
        )
        assert option.is_in_range(89.99, 0) is False  # Below lower
        assert option.is_in_range(110.01, 0) is False  # Above upper

    def test_is_in_range_reverse_mode(self):
        """Test is_in_range with reverse mode (pay outside range)."""
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_reverse=True,
        )
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            maturity=1.0,
            num_observations=12,
        )
        # Reverse: True when OUTSIDE range
        assert option.is_in_range(100.0, 0) is False  # Inside -> False
        assert option.is_in_range(89.0, 0) is True  # Outside -> True
        assert option.is_in_range(111.0, 0) is True  # Outside -> True

    def test_get_weighted_accrual_ratio(self, basic_range_config):
        """Test weighted accrual ratio calculation."""
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            num_observations=12,
        )
        # 9 out of 12 observations in range
        ratio = option.get_weighted_accrual_ratio(9.0, 12.0)
        assert ratio == 0.75

        # All in range
        ratio = option.get_weighted_accrual_ratio(12.0, 12.0)
        assert ratio == 1.0

        # None in range
        ratio = option.get_weighted_accrual_ratio(0.0, 12.0)
        assert ratio == 0.0

    def test_get_weighted_accrual_ratio_with_friday_weights(self, basic_range_config):
        """Test weighted accrual ratio with Friday=3 convention."""
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            num_observations=12,
        )
        # 5 business days: Mon(1), Tue(1), Wed(1), Thu(1), Fri(3) = total 7
        # If Mon, Wed, Fri in-range: in_range_weight = 1 + 1 + 3 = 5
        total_weights = 7.0
        in_range_weights = 5.0
        ratio = option.get_weighted_accrual_ratio(in_range_weights, total_weights)
        assert abs(ratio - 5.0 / 7.0) < 1e-10

    def test_get_payoff_100_percent(self, basic_range_config):
        """Test payoff when 100% of observations in range."""
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            num_observations=12,
            contract_multiplier=1.0,
        )
        # Payoff = initial_price * multiplier * rate * ratio * year_fraction
        # = 100 * 1 * 0.05 * 1.0 * 1.0 = 5.0
        payoff = option.get_payoff(spot=100.0, in_range_weights=12.0, total_weights=12.0)
        assert abs(payoff - 5.0) < 1e-10

    def test_get_payoff_50_percent(self, basic_range_config):
        """Test payoff when 50% of observations in range."""
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            num_observations=12,
            contract_multiplier=1.0,
        )
        # Payoff = 100 * 1 * 0.05 * 0.5 * 1.0 = 2.5
        payoff = option.get_payoff(spot=100.0, in_range_weights=6.0, total_weights=12.0)
        assert abs(payoff - 2.5) < 1e-10

    def test_get_payoff_0_percent(self, basic_range_config):
        """Test payoff when 0% of observations in range."""
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            num_observations=12,
            contract_multiplier=1.0,
        )
        payoff = option.get_payoff(spot=100.0, in_range_weights=0.0, total_weights=12.0)
        assert payoff == 0.0

    def test_get_payoff_with_contract_multiplier(self, basic_range_config):
        """Test payoff scales with contract_multiplier."""
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            num_observations=12,
            contract_multiplier=10000.0,  # Notional
        )
        # Payoff = 100 * 10000 * 0.05 * 1.0 * 1.0 = 50000
        payoff = option.get_payoff(spot=100.0, in_range_weights=12.0, total_weights=12.0)
        assert abs(payoff - 50000.0) < 1e-6


# =============================================================================
# Historical Observation Tests
# =============================================================================


class TestHistoricalObservations:
    """Tests for historical observation handling."""

    def test_resolve_observations_all_future(self, basic_pricing_env, basic_range_config):
        """Test resolving all future observations."""
        records = [
            RangeAccrualObservationRecord(observation_time=0.25),
            RangeAccrualObservationRecord(observation_time=0.5),
            RangeAccrualObservationRecord(observation_time=0.75),
            RangeAccrualObservationRecord(observation_time=1.0),
        ]
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            observation_records=records,
        )
        past, future, total_weights = option.resolve_observations(basic_pricing_env)
        assert len(past) == 0
        assert len(future) == 4
        assert total_weights == 4.0

    def test_resolve_observations_with_past(self, basic_pricing_env, basic_range_config):
        """Test resolving mix of past and future observations."""
        records = [
            RangeAccrualObservationRecord(
                observation_time=-0.25, weight=1.0, observed_in_range=True
            ),
            RangeAccrualObservationRecord(
                observation_time=-0.1, weight=1.0, observed_in_range=False
            ),
            RangeAccrualObservationRecord(observation_time=0.25, weight=1.0),
            RangeAccrualObservationRecord(observation_time=0.5, weight=1.0),
        ]
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            observation_records=records,
        )
        past, future, total_weights = option.resolve_observations(basic_pricing_env)
        assert len(past) == 2
        assert len(future) == 2
        assert total_weights == 4.0
        # Check past observations
        assert past[0] == (1.0, True)  # weight=1, in_range=True
        assert past[1] == (1.0, False)  # weight=1, in_range=False

    def test_has_past_observations_true(self, basic_pricing_env, basic_range_config):
        """Test has_past_observations returns True when past exists."""
        records = [
            RangeAccrualObservationRecord(
                observation_time=-0.1, weight=1.0, observed_in_range=True
            ),
            RangeAccrualObservationRecord(observation_time=0.5, weight=1.0),
        ]
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            observation_records=records,
        )
        assert option.has_past_observations(basic_pricing_env) is True

    def test_has_past_observations_false(self, basic_pricing_env, basic_range_config):
        """Test has_past_observations returns False when no past exists."""
        records = [
            RangeAccrualObservationRecord(observation_time=0.25, weight=1.0),
            RangeAccrualObservationRecord(observation_time=0.5, weight=1.0),
        ]
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            observation_records=records,
        )
        assert option.has_past_observations(basic_pricing_env) is False

    def test_get_past_accrual(self, basic_pricing_env, basic_range_config):
        """Test get_past_accrual calculation."""
        records = [
            RangeAccrualObservationRecord(
                observation_time=-0.3, weight=1.0, observed_in_range=True
            ),
            RangeAccrualObservationRecord(
                observation_time=-0.2, weight=1.0, observed_in_range=True
            ),
            RangeAccrualObservationRecord(
                observation_time=-0.1, weight=3.0, observed_in_range=False  # Friday
            ),
            RangeAccrualObservationRecord(observation_time=0.25, weight=1.0),
        ]
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            observation_records=records,
        )
        in_range_weights, past_total = option.get_past_accrual(basic_pricing_env)
        # Past observations: 1+1=2 in range, total weights = 1+1+3=5
        assert in_range_weights == 2.0
        assert past_total == 5.0

    def test_resolve_past_observation_missing_data(self, basic_pricing_env, basic_range_config):
        """Test error when past observation lacks observed_in_range."""
        records = [
            RangeAccrualObservationRecord(
                observation_time=-0.1, weight=1.0
                # Missing observed_in_range!
            ),
            RangeAccrualObservationRecord(observation_time=0.5, weight=1.0),
        ]
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            observation_records=records,
        )
        with pytest.raises(ValidationError, match="must have observed_in_range set"):
            option.resolve_observations(basic_pricing_env)

    def test_resolve_future_observation_with_data(self, basic_pricing_env, basic_range_config):
        """Test error when future observation has observed_in_range set."""
        records = [
            RangeAccrualObservationRecord(
                observation_time=0.5, weight=1.0, observed_in_range=True  # Should not have this!
            ),
        ]
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            observation_records=records,
        )
        with pytest.raises(ValidationError, match="should not have observed_in_range set"):
            option.resolve_observations(basic_pricing_env)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestAssignCalendarDayWeights:
    """Tests for assign_calendar_day_weights function."""

    def test_weekday_weights(self):
        """Test that weekdays get weight=1, Friday gets weight=3."""
        dates = [
            datetime(2025, 1, 6),  # Monday
            datetime(2025, 1, 7),  # Tuesday
            datetime(2025, 1, 8),  # Wednesday
            datetime(2025, 1, 9),  # Thursday
            datetime(2025, 1, 10),  # Friday
        ]
        weights = assign_calendar_day_weights(dates)
        assert weights == [1.0, 1.0, 1.0, 1.0, 3.0]

    def test_multiple_fridays(self):
        """Test multiple Fridays get weight=3."""
        dates = [
            datetime(2025, 1, 10),  # Friday
            datetime(2025, 1, 17),  # Friday
            datetime(2025, 1, 24),  # Friday
        ]
        weights = assign_calendar_day_weights(dates)
        assert weights == [3.0, 3.0, 3.0]


class TestGenerateRangeObservationRecords:
    """Tests for generate_range_observation_records function."""

    def test_daily_without_weights(self):
        """Test daily observations without calendar day weights."""
        records = generate_range_observation_records(
            start_date=datetime(2025, 1, 6),  # Monday
            end_date=datetime(2025, 1, 10),  # Friday
            frequency=ObservationFrequency.DAILY,
            use_calendar_day_weights=False,
        )
        assert len(records) == 5  # Mon-Fri
        assert all(r.weight == 1.0 for r in records)

    def test_daily_with_calendar_weights(self):
        """Test daily observations with calendar day weights."""
        records = generate_range_observation_records(
            start_date=datetime(2025, 1, 6),  # Monday
            end_date=datetime(2025, 1, 10),  # Friday
            frequency=ObservationFrequency.DAILY,
            use_calendar_day_weights=True,
        )
        assert len(records) == 5
        weights = [r.weight for r in records]
        assert weights == [1.0, 1.0, 1.0, 1.0, 3.0]  # Mon-Thu=1, Fri=3

    def test_skips_weekends(self):
        """Test that weekends are skipped."""
        records = generate_range_observation_records(
            start_date=datetime(2025, 1, 6),  # Monday
            end_date=datetime(2025, 1, 13),  # Following Monday
            frequency=ObservationFrequency.DAILY,
            use_calendar_day_weights=False,
        )
        # Mon-Fri (5) + Mon (1) = 6 business days
        assert len(records) == 6

    def test_with_per_observation_barriers(self):
        """Test generating records with per-observation barriers."""
        records = generate_range_observation_records(
            start_date=datetime(2025, 1, 6),
            end_date=datetime(2025, 1, 8),
            frequency=ObservationFrequency.DAILY,
            upper_barrier=115.0,
            lower_barrier=85.0,
        )
        assert len(records) == 3
        assert all(r.upper_barrier == 115.0 for r in records)
        assert all(r.lower_barrier == 85.0 for r in records)


class TestCreateStandardRangeAccrual:
    """Tests for create_standard_range_accrual factory function."""

    def test_basic_creation(self):
        """Test basic range accrual creation."""
        option = create_standard_range_accrual(
            initial_price=100.0,
            upper_barrier=110.0,
            lower_barrier=90.0,
            maturity=1.0,
            accrual_rate=0.05,
            num_observations=252,
        )
        assert option.initial_price == 100.0
        assert option.range_config.upper_barrier == 110.0
        assert option.range_config.lower_barrier == 90.0
        assert option.range_config.accrual_rate == 0.05
        assert option.range_config.is_reverse is False

    def test_with_contract_multiplier(self):
        """Test creation with contract multiplier."""
        option = create_standard_range_accrual(
            initial_price=100.0,
            upper_barrier=110.0,
            lower_barrier=90.0,
            maturity=1.0,
            contract_multiplier=10000.0,
        )
        assert option.contract_multiplier == 10000.0

    def test_invalid_barriers(self):
        """Test validation error for invalid barriers."""
        with pytest.raises(ValidationError, match="lower_barrier.*must be less than"):
            create_standard_range_accrual(
                initial_price=100.0,
                upper_barrier=90.0,  # Wrong
                lower_barrier=110.0,  # Wrong
                maturity=1.0,
            )


class TestCreateReverseRangeAccrual:
    """Tests for create_reverse_range_accrual factory function."""

    def test_reverse_flag(self):
        """Test reverse range accrual has is_reverse=True."""
        option = create_reverse_range_accrual(
            initial_price=100.0,
            upper_barrier=110.0,
            lower_barrier=90.0,
            maturity=1.0,
        )
        assert option.range_config.is_reverse is True


class TestCreateStepdownRangeAccrual:
    """Tests for create_stepdown_range_accrual factory function."""

    def test_stepdown_barriers(self):
        """Test step-down barriers narrow over time."""
        option = create_stepdown_range_accrual(
            initial_price=100.0,
            initial_upper_barrier=115.0,
            initial_lower_barrier=85.0,
            maturity=1.0,
            upper_stepdown_rate=0.01,  # -1% per period
            lower_stepdown_rate=0.01,  # +1% per period
            num_observations=12,
        )
        # Upper barrier list
        upper_barriers = option.range_config.upper_barrier
        assert isinstance(upper_barriers, list)
        assert upper_barriers[0] == 115.0  # First observation
        assert upper_barriers[-1] == 115.0 - 11 * 1.0  # 104.0 (12 obs, 11 steps)

        # Lower barrier list
        lower_barriers = option.range_config.lower_barrier
        assert isinstance(lower_barriers, list)
        assert lower_barriers[0] == 85.0  # First observation
        assert lower_barriers[-1] == 85.0 + 11 * 1.0  # 96.0 (12 obs, 11 steps)

    def test_stepdown_validation_crossing(self):
        """Test error when step-down causes barriers to cross."""
        with pytest.raises(ValidationError, match="Barriers cross"):
            create_stepdown_range_accrual(
                initial_price=100.0,
                initial_upper_barrier=105.0,  # Too close
                initial_lower_barrier=95.0,  # Too close
                maturity=1.0,
                upper_stepdown_rate=0.02,  # Aggressive step-down
                lower_stepdown_rate=0.02,  # Aggressive step-up
                num_observations=12,
            )


# =============================================================================
# Integration Tests
# =============================================================================


class TestRangeAccrualIntegration:
    """Integration tests for complete workflow."""

    def test_full_workflow_with_historical_data(self, basic_pricing_env):
        """Test complete workflow with partial historical observations."""
        # Create option with historical and future observations
        records = [
            # Past observations (already observed)
            RangeAccrualObservationRecord(
                observation_time=-0.25, weight=1.0, observed_in_range=True
            ),
            RangeAccrualObservationRecord(
                observation_time=-0.2, weight=1.0, observed_in_range=True
            ),
            RangeAccrualObservationRecord(
                observation_time=-0.15, weight=3.0, observed_in_range=False  # Friday
            ),
            RangeAccrualObservationRecord(
                observation_time=-0.1, weight=1.0, observed_in_range=True
            ),
            # Future observations
            RangeAccrualObservationRecord(observation_time=0.25, weight=1.0),
            RangeAccrualObservationRecord(observation_time=0.5, weight=1.0),
            RangeAccrualObservationRecord(observation_time=0.75, weight=3.0),  # Friday
            RangeAccrualObservationRecord(observation_time=1.0, weight=1.0),
        ]

        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.10,  # 10% annualized
            is_rate_annualized=True,
        )

        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            maturity=1.0,
            observation_records=records,
            contract_multiplier=1.0,
        )

        # Check past accrual
        in_range_weights, past_total = option.get_past_accrual(basic_pricing_env)
        # Past: 1+1+0+1 = 3 in-range weights (Friday was out)
        # Past total: 1+1+3+1 = 6
        assert in_range_weights == 3.0
        assert past_total == 6.0

        # Total weights
        total = option.get_total_weights()
        # All: 1+1+3+1+1+1+3+1 = 12
        assert total == 12.0

        # If all future observations are in-range:
        # Final in-range: 3 (past) + 1+1+3+1 (future) = 9
        # Ratio: 9/12 = 0.75
        final_in_range = 9.0
        payoff = option.get_payoff(
            spot=100.0,
            in_range_weights=final_in_range,
            total_weights=total,
        )
        # Payoff = 100 * 1 * 0.10 * 0.75 * 1.0 = 7.5
        assert abs(payoff - 7.5) < 1e-10

    def test_repr(self, basic_range_config):
        """Test string representation."""
        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=basic_range_config,
            maturity=1.0,
            num_observations=12,
        )
        repr_str = repr(option)
        assert "RangeAccrualOption" in repr_str
        assert "100.00" in repr_str
        assert "90.00" in repr_str
        assert "110.00" in repr_str
