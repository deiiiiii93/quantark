"""
Unit tests for Snowball option helper functions.
"""

import pytest
from asset.equity.product.option import (
    SnowballOption,
    create_standard_snowball,
    create_stepdown_snowball,
    create_european_ki_snowball,
    create_parachute_snowball,
    create_airbag_snowball,
    generate_ko_observation_dates,
    generate_stepdown_barriers,
)
from util.enum import ObservationType, CouponPayType, ProtectionType
from util.exceptions import ValidationError


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestGenerateKOObservationDates:
    """Tests for generate_ko_observation_dates utility."""

    def test_monthly_observations(self):
        """Test monthly observation dates for 1 year."""
        dates = generate_ko_observation_dates(1.0, "monthly")
        assert len(dates) == 12
        assert dates[0] == pytest.approx(1 / 12, rel=1e-6)
        assert dates[-1] == pytest.approx(1.0, rel=1e-6)

    def test_quarterly_observations(self):
        """Test quarterly observation dates for 1 year."""
        dates = generate_ko_observation_dates(1.0, "quarterly")
        assert len(dates) == 4
        assert dates == pytest.approx([0.25, 0.5, 0.75, 1.0], rel=1e-6)

    def test_weekly_observations(self):
        """Test weekly observation dates for 1 year."""
        dates = generate_ko_observation_dates(1.0, "weekly")
        assert len(dates) == 52
        assert dates[0] == pytest.approx(1 / 52, rel=1e-6)
        assert dates[-1] == pytest.approx(1.0, rel=1e-6)

    def test_skip_first_observations(self):
        """Test skipping first N observations (lock-out period)."""
        dates = generate_ko_observation_dates(1.0, "quarterly", skip_first=1)
        assert len(dates) == 3
        assert dates[0] == pytest.approx(0.5, rel=1e-6)

    def test_shorter_maturity(self):
        """Test with 6-month maturity."""
        dates = generate_ko_observation_dates(0.5, "monthly")
        assert len(dates) == 6
        assert dates[-1] == pytest.approx(0.5, rel=1e-6)

    def test_invalid_frequency(self):
        """Test that invalid frequency raises error."""
        with pytest.raises(ValidationError, match="frequency"):
            generate_ko_observation_dates(1.0, "biweekly")

    def test_negative_maturity(self):
        """Test that negative maturity raises error."""
        with pytest.raises(ValidationError, match="maturity"):
            generate_ko_observation_dates(-1.0, "monthly")

    def test_skip_all_observations(self):
        """Test that skipping all observations raises error."""
        with pytest.raises(ValidationError, match="skip_first"):
            generate_ko_observation_dates(1.0, "quarterly", skip_first=5)


class TestGenerateStepdownBarriers:
    """Tests for generate_stepdown_barriers utility."""

    def test_basic_stepdown(self):
        """Test basic step-down barrier generation."""
        barriers = generate_stepdown_barriers(103.0, 0.5, 4)
        assert barriers == pytest.approx([103.0, 102.5, 102.0, 101.5], rel=1e-6)

    def test_stepdown_with_floor(self):
        """Test step-down with minimum floor."""
        barriers = generate_stepdown_barriers(103.0, 2.0, 4, min_barrier=100.0)
        assert barriers == pytest.approx([103.0, 101.0, 100.0, 100.0], rel=1e-6)

    def test_no_stepdown(self):
        """Test with zero stepdown (flat barriers)."""
        barriers = generate_stepdown_barriers(103.0, 0.0, 4)
        assert all(b == 103.0 for b in barriers)

    def test_single_observation(self):
        """Test with single observation."""
        barriers = generate_stepdown_barriers(103.0, 0.5, 1)
        assert barriers == [103.0]

    def test_invalid_initial_barrier(self):
        """Test that non-positive initial barrier raises error."""
        with pytest.raises(ValidationError, match="initial_barrier"):
            generate_stepdown_barriers(0.0, 0.5, 4)

    def test_negative_stepdown(self):
        """Test that negative stepdown raises error."""
        with pytest.raises(ValidationError, match="stepdown_amount"):
            generate_stepdown_barriers(103.0, -0.5, 4)

    def test_zero_observations(self):
        """Test that zero observations raises error."""
        with pytest.raises(ValidationError, match="num_observations"):
            generate_stepdown_barriers(103.0, 0.5, 0)


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestCreateStandardSnowball:
    """Tests for create_standard_snowball helper."""

    def test_minimal_params(self):
        """Test with minimal required parameters."""
        snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )
        assert isinstance(snowball, SnowballOption)
        assert snowball.initial_price == 100.0
        assert snowball.strike == 100.0
        assert snowball.notional == 1_000_000.0

    def test_default_ko_barrier(self):
        """Test default KO barrier is 103% of initial price."""
        snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )
        assert snowball.barrier_config.ko_barrier == 103.0

    def test_default_ki_barrier(self):
        """Test default KI barrier is 75% of initial price."""
        snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )
        assert snowball.barrier_config.ki_barrier == 75.0

    def test_default_observations(self):
        """Test default 12 monthly observations."""
        snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )
        assert len(snowball.barrier_config.ko_observation_dates) == 12
        assert snowball.barrier_config.ko_observation_dates[-1] == pytest.approx(
            1.0, rel=1e-6
        )

    def test_continuous_ki(self):
        """Test KI is continuous by default."""
        snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )
        assert snowball.barrier_config.ki_continuous is True

    def test_custom_ko_barrier(self):
        """Test custom KO barrier override."""
        snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ko_barrier=105.0,
        )
        assert snowball.barrier_config.ko_barrier == 105.0

    def test_custom_notional(self):
        """Test custom notional."""
        snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            notional=5_000_000.0,
        )
        assert snowball.notional == 5_000_000.0

    def test_reverse_snowball(self):
        """Test reverse snowball creation."""
        snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            is_reverse=True,
        )
        assert snowball.is_reverse is True

    def test_kwargs_override_include_principal(self):
        """Test kwargs override for payoff config."""
        snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            include_principal=True,
        )
        assert snowball.payoff_config.include_principal is True

    def test_kwargs_override_coupon_pay_type(self):
        """Test kwargs override for accrual config."""
        snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            coupon_pay_type=CouponPayType.EXPIRY,
        )
        assert snowball.accrual_config.coupon_pay_type == CouponPayType.EXPIRY

    def test_invalid_initial_price(self):
        """Test validation rejects negative initial price."""
        with pytest.raises(ValidationError, match="initial_price"):
            create_standard_snowball(
                initial_price=-100.0,
                strike=100.0,
                maturity=1.0,
            )

    def test_invalid_maturity(self):
        """Test validation rejects zero maturity."""
        with pytest.raises(ValidationError, match="maturity"):
            create_standard_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=0.0,
            )


class TestCreateStepdownSnowball:
    """Tests for create_stepdown_snowball helper."""

    def test_basic_stepdown(self):
        """Test basic step-down snowball creation."""
        snowball = create_stepdown_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )
        assert isinstance(snowball, SnowballOption)
        # KO barrier should be a list
        assert isinstance(snowball.barrier_config.ko_barrier, list)
        assert len(snowball.barrier_config.ko_barrier) == 12

    def test_stepdown_decreasing(self):
        """Test that barriers decrease."""
        snowball = create_stepdown_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            stepdown_rate=0.005,
        )
        barriers = snowball.barrier_config.ko_barrier
        # Each barrier should be less than or equal to previous
        for i in range(1, len(barriers)):
            assert barriers[i] <= barriers[i - 1]

    def test_stepdown_first_barrier(self):
        """Test first barrier is initial_ko_barrier."""
        snowball = create_stepdown_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            initial_ko_barrier=105.0,
        )
        assert snowball.barrier_config.ko_barrier[0] == 105.0

    def test_custom_stepdown_rate(self):
        """Test custom stepdown rate."""
        snowball = create_stepdown_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            stepdown_rate=0.01,  # 1% per period
            num_observations=4,
        )
        barriers = snowball.barrier_config.ko_barrier
        # With 1% stepdown, barriers should decrease by 1.0 each period
        expected = [103.0, 102.0, 101.0, 100.0]
        assert barriers == pytest.approx(expected, rel=1e-6)


class TestCreateEuropeanKISnowball:
    """Tests for create_european_ki_snowball helper."""

    def test_single_ki_observation(self):
        """Test KI has single observation at maturity."""
        snowball = create_european_ki_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )
        assert snowball.barrier_config.ki_observation_dates == [1.0]

    def test_discrete_ki(self):
        """Test KI is discrete, not continuous."""
        snowball = create_european_ki_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )
        assert snowball.barrier_config.ki_continuous is False
        assert snowball.barrier_config.ki_observation_type == ObservationType.DISCRETE

    def test_multiple_ko_observations(self):
        """Test KO still has multiple observations."""
        snowball = create_european_ki_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            num_ko_observations=12,
        )
        assert len(snowball.barrier_config.ko_observation_dates) == 12


class TestCreateParachuteSnowball:
    """Tests for create_parachute_snowball helper."""

    def test_last_barrier_equals_ki(self):
        """Test last KO barrier equals KI barrier."""
        snowball = create_parachute_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )
        barriers = snowball.barrier_config.ko_barrier
        ki_barrier = snowball.barrier_config.ki_barrier
        assert barriers[-1] == ki_barrier

    def test_early_barriers_flat(self):
        """Test early barriers are flat at ko_barrier level."""
        snowball = create_parachute_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ko_barrier=105.0,
        )
        barriers = snowball.barrier_config.ko_barrier
        # All except last should be 105.0
        assert all(b == 105.0 for b in barriers[:-1])

    def test_custom_ki_barrier(self):
        """Test custom KI barrier (which becomes last KO barrier)."""
        snowball = create_parachute_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ko_barrier=105.0,
            ki_barrier=80.0,
        )
        barriers = snowball.barrier_config.ko_barrier
        assert barriers[-1] == 80.0
        assert snowball.barrier_config.ki_barrier == 80.0





class TestCreateAirbagSnowball:
    """Tests for create_airbag_snowball helper."""

    def test_participation_rate(self):
        """Test reduced participation rate is set."""
        snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            participation_rate=1.0,
            airbag_participation_rate=0.5,
            airbag_barrier=60.0,
        )
        assert snowball.payoff_config.participation_rate == 1.0
        assert snowball.airbag_config.airbag_participation_rate == 0.5
        assert snowball.airbag_config.airbag_barrier == 60.0

    def test_airbag_strike(self):
        """Test airbag strike is set correctly."""
        snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            airbag_strike=90.0,
        )
        assert snowball.airbag_config.airbag_strike == 90.0

    def test_default_participation(self):
        """Test default participation rate is 50%."""
        snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
        )
        assert snowball.airbag_config.airbag_participation_rate == 0.5

    def test_invalid_airbag_barrier(self):
        """Test airbag barrier must be less than KI barrier."""
        with pytest.raises(ValidationError, match="airbag_barrier"):
            create_airbag_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                ki_barrier=75.0,
                airbag_barrier=80.0,  # Invalid: airbag > ki
            )


# =============================================================================
# Integration Tests
# =============================================================================


class TestHelperIntegration:
    """Integration tests for helper functions."""

    def test_all_helpers_create_valid_snowballs(self):
        """Test all helpers create valid SnowballOption instances."""
        helpers = [
            create_standard_snowball,
            create_stepdown_snowball,
            create_european_ki_snowball,
            create_parachute_snowball,
            create_airbag_snowball,
        ]
        for helper in helpers:
            snowball = helper(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
            )
            assert isinstance(snowball, SnowballOption)
            # Should not raise validation error
            snowball.validate()

    def test_import_from_option_module(self):
        """Test helpers can be imported from option module."""
        from asset.equity.product.option import (
            create_standard_snowball,
            create_stepdown_snowball,
            create_european_ki_snowball,
            create_parachute_snowball,
            create_airbag_snowball,
            generate_ko_observation_dates,
            generate_stepdown_barriers,
        )

        # All should be callable
        assert callable(create_standard_snowball)
        assert callable(create_stepdown_snowball)
        assert callable(create_european_ki_snowball)
        assert callable(create_parachute_snowball)
        assert callable(create_airbag_snowball)
        assert callable(generate_ko_observation_dates)
        assert callable(generate_stepdown_barriers)
