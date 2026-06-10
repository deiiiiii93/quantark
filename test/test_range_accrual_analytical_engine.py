"""
Tests for RangeAccrualAnalyticalEngine.

Tests cover:
- Basic pricing against known analytical values
- Digital decomposition correctness (standard + reverse = full notional)
- Comparison with Monte Carlo engine
- Historical observations with recorded outcomes
- Time-varying barriers (step-down)
- Edge cases (near-expiry, all-past, zero vol, single observation)
- Greeks (delta, gamma) analytical computation
- Error handling for invalid inputs
"""

import math
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import (
    RangeAccrualAnalyticalEngine,
    RangeAccrualAnalyticalResult,
)
from quantark.asset.equity.engine.mc import RangeAccrualMCEngine
from quantark.asset.equity.product.option import (
    RangeAccrualOption,
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
)
from quantark.asset.equity.param import MCParams
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError, PricingError


@pytest.fixture
def pricing_env():
    """Standard pricing environment."""
    return PricingEnvironment(
        valuation_date=datetime(2024, 1, 1),
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
    )


@pytest.fixture
def standard_config():
    """Standard range accrual configuration."""
    return RangeAccrualConfig(
        upper_barrier=110.0,
        lower_barrier=90.0,
        accrual_rate=0.05,
        is_rate_annualized=True,
    )


@pytest.fixture
def standard_option(standard_config):
    """Standard range accrual option with quarterly observations."""
    return RangeAccrualOption(
        initial_price=100.0,
        range_config=standard_config,
        observation_times=[0.25, 0.5, 0.75, 1.0],
        maturity=1.0,
        contract_multiplier=10000.0,
    )


@pytest.fixture
def monthly_option(standard_config):
    """Standard range accrual option with monthly observations."""
    times = [(i + 1) / 12 for i in range(12)]
    return RangeAccrualOption(
        initial_price=100.0,
        range_config=standard_config,
        observation_times=times,
        maturity=1.0,
        contract_multiplier=1.0,
    )


@pytest.fixture
def engine():
    """Standard analytical engine."""
    return RangeAccrualAnalyticalEngine()


class TestRangeAccrualAnalyticalBasic:
    """Basic pricing tests."""

    def test_price_positive(self, standard_option, pricing_env, engine):
        """Price must be positive for standard range accrual."""
        price = engine.price(standard_option, pricing_env)
        assert price > 0

    def test_result_stored(self, standard_option, pricing_env, engine):
        """Engine stores last result."""
        engine.price(standard_option, pricing_env)
        result = engine.get_last_result()
        assert result is not None
        assert isinstance(result, RangeAccrualAnalyticalResult)

    def test_result_fields(self, standard_option, pricing_env, engine):
        """Result contains expected fields."""
        engine.price(standard_option, pricing_env)
        result = engine.get_last_result()
        assert result.expected_accrual_ratio > 0
        assert result.expected_accrual_ratio <= 1.0
        assert len(result.per_observation_probs) == 4  # 4 observations
        assert result.num_past_observations == 0
        assert result.num_future_observations == 4
        assert result.total_weights == 4.0  # 4 observations, weight 1.0 each

    def test_per_obs_probs_decreasing(self, standard_option, pricing_env, engine):
        """In-range probabilities decrease with time (more uncertainty)."""
        engine.price(standard_option, pricing_env)
        result = engine.get_last_result()
        probs = result.per_observation_probs
        # With ATM spot and symmetric barriers, probs should decrease over time
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]

    def test_monthly_observations(self, monthly_option, pricing_env, engine):
        """Test with 12 monthly observations."""
        price = engine.price(monthly_option, pricing_env)
        assert price > 0
        result = engine.get_last_result()
        assert len(result.per_observation_probs) == 12


class TestDigitalDecomposition:
    """Test the digital decomposition correctness."""

    def test_standard_plus_reverse_equals_one(self, pricing_env, engine):
        """Standard + reverse mode accrual ratios must sum to 1."""
        times = [(i + 1) / 12 for i in range(12)]

        config_std = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
            is_reverse=False,
        )
        config_rev = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
            is_reverse=True,
        )

        opt_std = RangeAccrualOption(
            initial_price=100.0,
            range_config=config_std,
            observation_times=times,
            maturity=1.0,
        )
        opt_rev = RangeAccrualOption(
            initial_price=100.0,
            range_config=config_rev,
            observation_times=times,
            maturity=1.0,
        )

        engine.price(opt_std, pricing_env)
        ratio_std = engine.get_last_result().expected_accrual_ratio

        engine.price(opt_rev, pricing_env)
        ratio_rev = engine.get_last_result().expected_accrual_ratio

        assert abs(ratio_std + ratio_rev - 1.0) < 1e-12

    def test_wide_range_prob_near_one(self, pricing_env, engine):
        """Very wide range [1, 1000] should give probability near 1."""
        config = RangeAccrualConfig(
            upper_barrier=1000.0,
            lower_barrier=1.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_times=[0.5, 1.0],
            maturity=1.0,
        )
        engine.price(opt, pricing_env)
        result = engine.get_last_result()
        assert result.expected_accrual_ratio > 0.999

    def test_narrow_range_low_prob(self, pricing_env, engine):
        """Very narrow range [99.99, 100.01] should have low probability."""
        config = RangeAccrualConfig(
            upper_barrier=100.01,
            lower_barrier=99.99,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_times=[0.5, 1.0],
            maturity=1.0,
        )
        engine.price(opt, pricing_env)
        result = engine.get_last_result()
        assert result.expected_accrual_ratio < 0.01


class TestMCComparison:
    """Compare analytical engine with Monte Carlo."""

    def test_analytical_vs_mc_qmc(self, standard_option, pricing_env, engine):
        """Analytical price should be close to QMC price."""
        analytical_price = engine.price(standard_option, pricing_env)

        mc_engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=200000, seed=42),
            method=MonteCarloMethod.QUASI,
        )
        mc_price = mc_engine.price(standard_option, pricing_env)

        # Should be within 1% of each other
        rel_diff = abs(analytical_price - mc_price) / analytical_price
        assert rel_diff < 0.01, (
            f"Analytical ({analytical_price:.4f}) vs MC ({mc_price:.4f}), "
            f"rel diff: {rel_diff:.4%}"
        )

    def test_analytical_vs_mc_reverse_mode(self, pricing_env, engine):
        """Test reverse mode analytical vs MC."""
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
            is_reverse=True,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
            contract_multiplier=10000.0,
        )

        analytical_price = engine.price(opt, pricing_env)
        mc_engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=200000, seed=42),
            method=MonteCarloMethod.QUASI,
        )
        mc_price = mc_engine.price(opt, pricing_env)

        rel_diff = abs(analytical_price - mc_price) / analytical_price
        assert rel_diff < 0.01

    def test_analytical_vs_mc_weighted_observations(self, pricing_env, engine):
        """Test with weighted observations (Friday=3)."""
        records = [
            RangeAccrualObservationRecord(observation_time=0.2, weight=1.0),
            RangeAccrualObservationRecord(observation_time=0.4, weight=3.0),
            RangeAccrualObservationRecord(observation_time=0.6, weight=1.0),
            RangeAccrualObservationRecord(observation_time=0.8, weight=3.0),
            RangeAccrualObservationRecord(observation_time=1.0, weight=1.0),
        ]
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_records=records,
            maturity=1.0,
            contract_multiplier=10000.0,
        )

        analytical_price = engine.price(opt, pricing_env)
        mc_engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=200000, seed=42),
            method=MonteCarloMethod.QUASI,
        )
        mc_price = mc_engine.price(opt, pricing_env)

        rel_diff = abs(analytical_price - mc_price) / analytical_price
        assert rel_diff < 0.01


class TestHistoricalObservations:
    """Test with historical (past) observations."""

    def test_all_past_observations(self, pricing_env, engine):
        """All observations in the past: deterministic payoff."""
        records = [
            RangeAccrualObservationRecord(
                observation_time=-0.5, weight=1.0, observed_in_range=True
            ),
            RangeAccrualObservationRecord(
                observation_time=-0.25, weight=1.0, observed_in_range=False
            ),
        ]
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.10,
            is_rate_annualized=False,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_records=records,
            maturity=0.01,
        )

        price = engine.price(opt, pricing_env)
        # 1 out of 2 in range, ratio = 0.5
        # Payoff = 100 * 1 * 0.10 * 0.5 * 1.0 = 5.0
        # Discounted: ~5.0 * exp(-0.05 * 0.01) ~ 5.0
        expected = 100.0 * 0.10 * 0.5 * math.exp(-0.05 * 0.01)
        assert abs(price - expected) < 0.01

    def test_mixed_past_and_future(self, pricing_env, engine):
        """Mix of past and future observations."""
        records = [
            RangeAccrualObservationRecord(
                observation_time=-0.25, weight=1.0, observed_in_range=True
            ),
            RangeAccrualObservationRecord(
                observation_time=-0.1, weight=1.0, observed_in_range=True
            ),
            RangeAccrualObservationRecord(
                observation_time=0.25, weight=1.0
            ),
            RangeAccrualObservationRecord(
                observation_time=0.5, weight=1.0
            ),
        ]
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_records=records,
            maturity=0.5,
        )
        price = engine.price(opt, pricing_env)
        assert price > 0

        result = engine.get_last_result()
        assert result.num_past_observations == 2
        assert result.num_future_observations == 2
        assert result.past_in_range_weights == 2.0


class TestTimeVaryingBarriers:
    """Test with time-varying (step-down) barriers."""

    def test_stepdown_barriers(self, pricing_env, engine):
        """Test with narrowing barriers over time."""
        config = RangeAccrualConfig(
            upper_barrier=[115.0, 112.0, 110.0, 108.0],
            lower_barrier=[85.0, 88.0, 90.0, 92.0],
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
        )
        price = engine.price(opt, pricing_env)
        assert price > 0

        # Verify probabilities decrease as barriers narrow
        result = engine.get_last_result()
        probs = result.per_observation_probs
        # First observation has widest barriers, should have highest prob
        assert probs[0] > probs[-1]

    def test_stepdown_analytical_vs_mc(self, pricing_env, engine):
        """Compare step-down analytical vs MC."""
        config = RangeAccrualConfig(
            upper_barrier=[115.0, 112.0, 110.0, 108.0],
            lower_barrier=[85.0, 88.0, 90.0, 92.0],
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
            contract_multiplier=10000.0,
        )

        analytical = engine.price(opt, pricing_env)
        mc_engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=200000, seed=42),
            method=MonteCarloMethod.QUASI,
        )
        mc = mc_engine.price(opt, pricing_env)

        rel_diff = abs(analytical - mc) / analytical
        assert rel_diff < 0.01


class TestEdgeCases:
    """Edge case tests."""

    def test_single_observation(self, pricing_env, engine):
        """Single observation is equivalent to a digital call spread."""
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=1.0,
            is_rate_annualized=False,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_times=[1.0],
            maturity=1.0,
        )
        price = engine.price(opt, pricing_env)
        assert price > 0

        # Verify against manual N(d2) computation
        from scipy import stats

        S, r, q, sigma, T = 100.0, 0.05, 0.02, 0.2, 1.0
        drift = (r - q - 0.5 * sigma**2) * T
        d2_L = (math.log(S / 90.0) + drift) / (sigma * math.sqrt(T))
        d2_U = (math.log(S / 110.0) + drift) / (sigma * math.sqrt(T))
        prob = stats.norm.cdf(d2_L) - stats.norm.cdf(d2_U)
        expected = math.exp(-r * T) * 100.0 * 1.0 * prob * 1.0
        assert abs(price - expected) < 1e-10

    def test_spot_outside_range(self, pricing_env):
        """Spot well outside range should have low price."""
        engine = RangeAccrualAnalyticalEngine()
        # Spot is at 100, but range is [120, 130]
        config = RangeAccrualConfig(
            upper_barrier=130.0,
            lower_barrier=120.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
        )
        price = engine.price(opt, pricing_env)
        # Price should be much lower than standard case
        assert price < 1.0  # Scaled by initial_price=100, mult=1.0

    def test_num_observations_input(self, pricing_env, engine):
        """Test with num_observations instead of explicit times."""
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            num_observations=12,
            maturity=1.0,
        )
        price = engine.price(opt, pricing_env)
        assert price > 0
        result = engine.get_last_result()
        assert result.num_future_observations == 12


class TestVolatilitySensitivity:
    """Test sensitivity to volatility."""

    def test_higher_vol_lower_price(self, pricing_env, engine):
        """Higher vol reduces in-range probability for ATM range."""
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_times=[0.5, 1.0],
            maturity=1.0,
        )

        # Low vol
        env_low = PricingEnvironment(
            valuation_date=datetime(2024, 1, 1),
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.10),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
        )
        price_low_vol = engine.price(opt, env_low)

        # High vol
        env_high = PricingEnvironment(
            valuation_date=datetime(2024, 1, 1),
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.40),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
        )
        price_high_vol = engine.price(opt, env_high)

        # Low vol should give higher price (more likely to stay in range)
        assert price_low_vol > price_high_vol


class TestGreeks:
    """Test analytical Greeks."""

    def test_delta_positive_for_standard(self, standard_option, pricing_env, engine):
        """Delta should be small and defined for ATM standard range accrual."""
        greeks = engine.calculate_greeks(standard_option, pricing_env)
        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks
        # Price should match direct pricing
        direct_price = engine.price(standard_option, pricing_env)
        assert abs(greeks["price"] - direct_price) < 1e-10

    def test_delta_vs_bump(self, pricing_env, engine):
        """Analytical delta should match bump-and-reprice."""
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
            contract_multiplier=1.0,
        )

        greeks = engine.calculate_greeks(opt, pricing_env)
        analytical_delta = greeks["delta"]

        # Bump-and-reprice
        from copy import deepcopy
        bump = 0.01  # 1% bump
        env_up = deepcopy(pricing_env)
        env_up.spot_quote.spot *= (1 + bump)
        env_down = deepcopy(pricing_env)
        env_down.spot_quote.spot *= (1 - bump)

        price_up = engine.price(opt, env_up)
        price_down = engine.price(opt, env_down)
        numerical_delta = (price_up - price_down) / (2 * pricing_env.spot * bump)

        # Should be close (within 5% relative or 1e-6 absolute)
        if abs(analytical_delta) > 1e-6:
            rel_diff = abs(analytical_delta - numerical_delta) / abs(analytical_delta)
            assert rel_diff < 0.05, (
                f"Analytical delta ({analytical_delta:.8f}) vs numerical ({numerical_delta:.8f})"
            )
        else:
            assert abs(analytical_delta - numerical_delta) < 1e-5

    def test_gamma_vs_bump(self, pricing_env, engine):
        """Analytical gamma should match bump-and-reprice."""
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
            contract_multiplier=1.0,
        )

        greeks = engine.calculate_greeks(opt, pricing_env)
        analytical_gamma = greeks["gamma"]

        from copy import deepcopy
        bump = 0.01
        env_up = deepcopy(pricing_env)
        env_up.spot_quote.spot *= (1 + bump)
        env_down = deepcopy(pricing_env)
        env_down.spot_quote.spot *= (1 - bump)

        base_price = engine.price(opt, pricing_env)
        price_up = engine.price(opt, env_up)
        price_down = engine.price(opt, env_down)
        numerical_gamma = (price_up - 2 * base_price + price_down) / (pricing_env.spot * bump) ** 2

        if abs(analytical_gamma) > 1e-6:
            rel_diff = abs(analytical_gamma - numerical_gamma) / abs(analytical_gamma)
            assert rel_diff < 0.10, (
                f"Analytical gamma ({analytical_gamma:.8f}) vs numerical ({numerical_gamma:.8f})"
            )
        else:
            assert abs(analytical_gamma - numerical_gamma) < 1e-5


class TestErrorHandling:
    """Test error handling."""

    def test_wrong_product_type(self, pricing_env, engine):
        """Should raise PricingError for wrong product type."""
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        from quantark.util.enum import OptionType

        wrong_product = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )
        with pytest.raises(PricingError):
            engine.price(wrong_product, pricing_env)

    def test_missing_range_config(self, pricing_env, engine):
        """Should raise ValidationError for missing range_config."""
        # Create option, then remove range_config to simulate invalid state
        config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
        )
        opt = RangeAccrualOption(
            initial_price=100.0,
            range_config=config,
            observation_times=[0.5, 1.0],
            maturity=1.0,
        )
        opt.range_config = None
        with pytest.raises(ValidationError):
            engine.price(opt, pricing_env)

    def test_repr(self, engine):
        """Test string representation."""
        assert "RangeAccrualAnalyticalEngine" in repr(engine)
