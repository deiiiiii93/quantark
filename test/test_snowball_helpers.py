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
        assert snowball.contract_multiplier == 1.0

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

    def test_custom_contract_multiplier(self):
        """Test custom contract multiplier."""
        snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
        )
        assert snowball.contract_multiplier == 10_000.0

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


class TestAirbagPayoffCalculation:
    """Tests for airbag payoff calculation in get_maturity_payoff_v1."""

    def test_airbag_payoff_below_airbag_barrier(self):
        """Test airbag participation rate is used when spot < airbag_barrier."""
        snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            airbag_barrier=60.0,
            airbag_participation_rate=0.5,
            participation_rate=1.0,
            include_principal=False,
        )
        # Spot = 50, below airbag_barrier = 60
        # Standard snowball V1: downside = participation * min(spot - strike, 0) * N / S0
        payoff = snowball.get_maturity_payoff_v1(spot=50.0)
        expected = 0.5 * (50.0 - 100.0) * snowball.contract_multiplier
        assert payoff == pytest.approx(expected, rel=1e-6)

    def test_standard_payoff_above_airbag_barrier(self):
        """Test standard participation rate is used when spot >= airbag_barrier."""
        snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            airbag_barrier=60.0,
            airbag_participation_rate=0.5,
            participation_rate=1.0,
            include_principal=False,
        )
        # Spot = 70, above airbag_barrier = 60
        payoff = snowball.get_maturity_payoff_v1(spot=70.0)
        expected = 1.0 * (70.0 - 100.0) * snowball.contract_multiplier
        assert payoff == pytest.approx(expected, rel=1e-6)

    def test_airbag_reduces_loss(self):
        """Test airbag payoff results in smaller loss than standard."""
        # Create airbag snowball
        airbag_snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            airbag_barrier=60.0,
            airbag_participation_rate=0.5,
            participation_rate=1.0,
            include_principal=False,
        )
        # Create standard snowball with same params (no airbag)
        standard_snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            participation_rate=1.0,
            include_principal=False,
        )
        
        # Test at spot=50 (below airbag barrier)
        airbag_payoff = airbag_snowball.get_maturity_payoff_v1(spot=50.0)
        standard_payoff = standard_snowball.get_maturity_payoff_v1(spot=50.0)
        
        # Airbag should have smaller loss (higher payoff since both are negative)
        assert airbag_payoff > standard_payoff
        # Specifically, airbag loss should be 50% of standard loss
        assert airbag_payoff == pytest.approx(standard_payoff * 0.5, rel=1e-6)

    def test_airbag_with_custom_strike(self):
        """Test airbag payoff uses airbag_strike when specified."""
        snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            airbag_barrier=60.0,
            airbag_participation_rate=0.5,
            airbag_strike=90.0,  # Custom airbag strike
            include_principal=False,
        )
        # Spot = 50, below airbag_barrier = 60
        payoff = snowball.get_maturity_payoff_v1(spot=50.0)
        expected = 0.5 * (50.0 - 90.0) * snowball.contract_multiplier
        assert payoff == pytest.approx(expected, rel=1e-6)

    def test_airbag_with_principal(self):
        """Test airbag payoff includes principal when configured."""
        snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            airbag_barrier=60.0,
            airbag_participation_rate=0.5,
            include_principal=True,
        )
        # Spot = 50, below airbag_barrier = 60
        payoff = snowball.get_maturity_payoff_v1(spot=50.0)
        principal = snowball.initial_price * snowball.contract_multiplier
        downside = 0.5 * (50.0 - 100.0) * snowball.contract_multiplier
        expected = principal + downside
        assert payoff == pytest.approx(expected, rel=1e-6)

    def test_airbag_at_barrier_boundary(self):
        """Test behavior when spot equals airbag barrier exactly."""
        snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            airbag_barrier=60.0,
            airbag_participation_rate=0.5,
            participation_rate=1.0,
            include_principal=False,
        )
        # Spot = 60, exactly at airbag_barrier
        # Standard payoff applies (spot >= airbag_barrier, strict inequality for airbag)
        payoff = snowball.get_maturity_payoff_v1(spot=60.0)
        expected = 1.0 * (60.0 - 100.0) * snowball.contract_multiplier
        assert payoff == pytest.approx(expected, rel=1e-6)

    def test_airbag_reverse_snowball(self):
        """Test airbag payoff with reverse snowball."""
        snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=125.0,  # KI is up for reverse
            airbag_barrier=140.0,  # Airbag is up for reverse
            airbag_participation_rate=0.5,
            participation_rate=1.0,
            is_reverse=True,
            include_principal=False,
        )
        # Reverse snowball: loss when spot > strike
        # Airbag applies when spot > airbag_barrier
        # Spot = 150, above airbag_barrier = 140
        payoff = snowball.get_maturity_payoff_v1(spot=150.0)
        expected = 0.5 * (100.0 - 150.0) * snowball.contract_multiplier
        assert payoff == pytest.approx(expected, rel=1e-6)

    def test_no_airbag_config(self):
        """Test standard snowball without airbag behaves as before."""
        snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            participation_rate=1.0,
            include_principal=False,
        )
        # No airbag config, standard payoff at any spot
        payoff = snowball.get_maturity_payoff_v1(spot=50.0)
        expected = 1.0 * (50.0 - 100.0) * snowball.contract_multiplier
        assert payoff == pytest.approx(expected, rel=1e-6)


# =============================================================================
# Integration Tests
# =============================================================================


class TestAirbagMCEngineIntegration:
    """Integration tests for airbag snowball with MC engine."""

    def test_airbag_snowball_mc_pricing(self):
        """Test MC engine correctly prices airbag snowball."""
        from datetime import datetime
        from asset.equity.engine.mc import SnowballMCEngine
        from asset.equity.param import MCParams
        from param import SpotQuote, FlatVolSurface, FlatRateCurve
        from priceenv import PricingEnvironment
        
        # Create pricing environment (volatility will likely trigger KI for V1 payoff testing)
        pricing_env = PricingEnvironment(
            valuation_date=datetime(2024, 1, 1),
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.30),
            rate_curve=FlatRateCurve(rate=0.05),
        )
        
        # Create airbag snowball with include_principal=True for positive payoffs
        airbag_snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            airbag_barrier=60.0,
            airbag_participation_rate=0.5,
            participation_rate=1.0,
            include_principal=True,
        )

        # Create standard snowball with same params
        standard_snowball = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            participation_rate=1.0,
            include_principal=True,
        )
        
        # Price both with MC engine (use fewer paths for speed in tests)
        engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42))
        
        airbag_price = engine.price(airbag_snowball, pricing_env)
        standard_price = engine.price(standard_snowball, pricing_env)
        
        # Airbag snowball should be worth more (less downside loss)
        assert airbag_price > standard_price

        # Both prices should be positive (with include_principal=True)
        assert airbag_price > 0
        assert standard_price > 0

    def test_airbag_mc_statistics(self):
        """Test MC engine returns reasonable statistics for airbag snowball."""
        from datetime import datetime
        from asset.equity.engine.mc import SnowballMCEngine
        from asset.equity.param import MCParams
        from param import SpotQuote, FlatVolSurface, FlatRateCurve
        from priceenv import PricingEnvironment
        
        pricing_env = PricingEnvironment(
            valuation_date=datetime(2024, 1, 1),
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.30),
            rate_curve=FlatRateCurve(rate=0.05),
        )
        
        snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            include_principal=True,
        )

        engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42))
        engine.price(snowball, pricing_env)
        
        result = engine.get_last_result()
        
        # Probabilities should sum to 1
        total_prob = result.ko_probability + result.v0_probability + result.v1_probability
        assert total_prob == pytest.approx(1.0, rel=1e-6)
        
        # All probabilities should be in [0, 1]
        assert 0 <= result.ko_probability <= 1
        assert 0 <= result.v0_probability <= 1
        assert 0 <= result.v1_probability <= 1


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
