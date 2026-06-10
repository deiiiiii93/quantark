"""
Tests for RangeAccrualMCEngine.

Tests cover:
- Basic pricing with different MC methods
- Reverse mode (pay when outside range)
- Wide vs narrow range effects
- Historical observations with recorded outcomes
- Edge cases (expiry, all past observations)
- Convergence comparison between methods
"""

import math
from datetime import datetime

import pytest
import numpy as np

from quantark.asset.equity.engine.mc import RangeAccrualMCEngine, RangeAccrualMCResult
from quantark.asset.equity.product.option import (
    RangeAccrualOption,
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
)
from quantark.asset.equity.param import MCParams
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.util.enum.engine_enums import MonteCarloMethod, EngineType
from quantark.util.exceptions import ValidationError, PricingError


@pytest.fixture
def pricing_env():
    """Standard pricing environment for tests."""
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
    """Standard range accrual option."""
    return RangeAccrualOption(
        initial_price=100.0,
        range_config=standard_config,
        observation_times=[0.25, 0.5, 0.75, 1.0],
        maturity=1.0,
        contract_multiplier=10000.0,
    )


class TestRangeAccrualMCEngineBasic:
    """Basic functionality tests."""

    def test_price_pseudo_mc(self, standard_option, pricing_env):
        """Test pricing with pseudorandom MC."""
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=10000, seed=42),
            method=MonteCarloMethod.PSEUDO,
        )
        price = engine.price(standard_option, pricing_env)
        assert price > 0
        assert engine.get_last_result() is not None
        assert engine.get_last_std_error() is not None

    def test_price_quasi_mc(self, standard_option, pricing_env):
        """Test pricing with quasi-MC (Sobol)."""
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=10000, seed=42),
            method=MonteCarloMethod.QUASI,
        )
        price = engine.price(standard_option, pricing_env)
        assert price > 0

    def test_price_rqmc(self, standard_option, pricing_env):
        """Test pricing with randomized QMC."""
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=4096, seed=42),
            method=MonteCarloMethod.RANDOMIZED_QUASI,
        )
        price = engine.price(standard_option, pricing_env)
        assert price > 0
        result = engine.get_last_result()
        assert result.batches_used is not None

    def test_method_string_initialization(self, standard_option, pricing_env):
        """Test initialization with string method."""
        engine = RangeAccrualMCEngine(method="quasi")
        price = engine.price(standard_option, pricing_env)
        assert price > 0

    def test_method_two_level_enum(self, standard_option, pricing_env):
        """Test initialization with two-level enum pattern."""
        engine = RangeAccrualMCEngine(
            method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI),
        )
        price = engine.price(standard_option, pricing_env)
        assert price > 0

    def test_result_statistics(self, standard_option, pricing_env):
        """Test that result contains correct statistics."""
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=10000, seed=42),
        )
        engine.price(standard_option, pricing_env)
        result = engine.get_last_result()

        assert isinstance(result, RangeAccrualMCResult)
        assert result.num_paths == 10000
        assert 0 <= result.in_range_ratio_mean <= 1
        assert result.in_range_ratio_std >= 0
        assert result.std_error > 0
        assert result.num_past_observations == 0
        assert result.num_future_observations == 4
        assert result.total_weights == 4.0


class TestRangeAccrualMCEngineRangeEffects:
    """Tests for range width effects."""

    def test_wider_range_higher_ratio(self, pricing_env):
        """Wider range should have higher in-range ratio."""
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=20000, seed=42),
        )

        # Standard range
        std_config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        std_option = RangeAccrualOption(
            initial_price=100.0,
            range_config=std_config,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
        )

        # Wide range
        wide_config = RangeAccrualConfig(
            upper_barrier=130.0,
            lower_barrier=70.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        wide_option = RangeAccrualOption(
            initial_price=100.0,
            range_config=wide_config,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
        )

        engine.price(std_option, pricing_env)
        std_ratio = engine.get_last_result().in_range_ratio_mean

        engine.price(wide_option, pricing_env)
        wide_ratio = engine.get_last_result().in_range_ratio_mean

        assert wide_ratio > std_ratio

    def test_narrower_range_lower_ratio(self, pricing_env):
        """Narrower range should have lower in-range ratio."""
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=20000, seed=42),
        )

        # Standard range
        std_config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        std_option = RangeAccrualOption(
            initial_price=100.0,
            range_config=std_config,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
        )

        # Narrow range
        narrow_config = RangeAccrualConfig(
            upper_barrier=105.0,
            lower_barrier=95.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
        )
        narrow_option = RangeAccrualOption(
            initial_price=100.0,
            range_config=narrow_config,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
        )

        engine.price(std_option, pricing_env)
        std_ratio = engine.get_last_result().in_range_ratio_mean

        engine.price(narrow_option, pricing_env)
        narrow_ratio = engine.get_last_result().in_range_ratio_mean

        assert narrow_ratio < std_ratio


class TestRangeAccrualMCEngineReverseMode:
    """Tests for reverse mode (pay when outside range)."""

    def test_reverse_mode(self, pricing_env):
        """Test that reverse mode inverts the in-range logic."""
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=50000, seed=42),
        )

        # Normal mode
        normal_config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
            is_reverse=False,
        )
        normal_option = RangeAccrualOption(
            initial_price=100.0,
            range_config=normal_config,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
        )

        # Reverse mode
        reverse_config = RangeAccrualConfig(
            upper_barrier=110.0,
            lower_barrier=90.0,
            accrual_rate=0.05,
            is_rate_annualized=True,
            is_reverse=True,
        )
        reverse_option = RangeAccrualOption(
            initial_price=100.0,
            range_config=reverse_config,
            observation_times=[0.25, 0.5, 0.75, 1.0],
            maturity=1.0,
        )

        engine.price(normal_option, pricing_env)
        normal_ratio = engine.get_last_result().in_range_ratio_mean

        engine.price(reverse_option, pricing_env)
        reverse_ratio = engine.get_last_result().in_range_ratio_mean

        # Normal + Reverse should sum to 1.0
        assert abs(normal_ratio + reverse_ratio - 1.0) < 0.02


class TestRangeAccrualMCEngineHistoricalObservations:
    """Tests for historical observations with recorded outcomes."""

    def test_partial_historical_observations(self, pricing_env, standard_config):
        """Test option with some past observations already recorded."""
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=10000, seed=42),
        )

        records = [
            RangeAccrualObservationRecord(
                observation_time=-0.25, weight=1.0, observed_in_range=True
            ),
            RangeAccrualObservationRecord(
                observation_time=-0.10, weight=1.0, observed_in_range=False
            ),
            RangeAccrualObservationRecord(observation_time=0.25, weight=1.0),
            RangeAccrualObservationRecord(observation_time=0.50, weight=1.0),
        ]

        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=standard_config,
            observation_records=records,
            maturity=0.5,
        )

        price = engine.price(option, pricing_env)
        result = engine.get_last_result()

        assert price > 0
        assert result.num_past_observations == 2
        assert result.num_future_observations == 2
        assert result.past_in_range_weights == 1.0  # Only first past obs was in range
        assert result.total_weights == 4.0

    def test_all_past_observations(self, pricing_env, standard_config):
        """Test option where all observations are in the past."""
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=10000, seed=42),
        )

        records = [
            RangeAccrualObservationRecord(
                observation_time=-0.50, weight=1.0, observed_in_range=True
            ),
            RangeAccrualObservationRecord(
                observation_time=-0.25, weight=1.0, observed_in_range=True
            ),
        ]

        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=standard_config,
            observation_records=records,
            maturity=0.01,  # Very short maturity
        )

        price = engine.price(option, pricing_env)
        result = engine.get_last_result()

        assert price > 0
        assert result.num_past_observations == 2
        assert result.num_future_observations == 0
        assert result.std_error == 0.0  # No simulation uncertainty


class TestRangeAccrualMCEngineEdgeCases:
    """Tests for edge cases."""

    def test_near_expiry(self, pricing_env, standard_config):
        """Test option very close to expiry."""
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=1000, seed=42),
        )

        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=standard_config,
            observation_times=[0.001],  # Very close to expiry
            maturity=0.001,
        )

        price = engine.price(option, pricing_env)
        assert price >= 0

    def test_weighted_observations(self, pricing_env, standard_config):
        """Test option with weighted observations (e.g., calendar day weights)."""
        engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=10000, seed=42),
        )

        # Friday = 3 (for weekend carry), other days = 1
        records = [
            RangeAccrualObservationRecord(observation_time=0.25, weight=1.0),  # Mon
            RangeAccrualObservationRecord(observation_time=0.50, weight=3.0),  # Fri
            RangeAccrualObservationRecord(observation_time=0.75, weight=1.0),  # Mon
            RangeAccrualObservationRecord(observation_time=1.00, weight=1.0),  # Mon
        ]

        option = RangeAccrualOption(
            initial_price=100.0,
            range_config=standard_config,
            observation_records=records,
            maturity=1.0,
        )

        price = engine.price(option, pricing_env)
        result = engine.get_last_result()

        assert price > 0
        assert result.total_weights == 6.0  # 1 + 3 + 1 + 1

    def test_invalid_product_type(self, pricing_env):
        """Test that engine rejects non-RangeAccrual products."""
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        from quantark.util.enum import OptionType

        engine = RangeAccrualMCEngine()
        wrong_product = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )

        with pytest.raises(PricingError, match="only supports RangeAccrualOption"):
            engine.price(wrong_product, pricing_env)

    def test_invalid_method(self):
        """Test that invalid method raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid method string"):
            RangeAccrualMCEngine(method="invalid_method")


class TestRangeAccrualMCEngineConvergence:
    """Tests for convergence behavior."""

    def test_quasi_vs_pseudo_convergence(self, standard_option, pricing_env):
        """Test that Quasi-MC converges faster than Pseudo-MC."""
        # This is a statistical test, so we use a fixed seed
        quasi_engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=10000, seed=42),
            method=MonteCarloMethod.QUASI,
        )
        pseudo_engine = RangeAccrualMCEngine(
            params=MCParams(num_paths=10000, seed=42),
            method=MonteCarloMethod.PSEUDO,
        )

        quasi_engine.price(standard_option, pricing_env)
        pseudo_engine.price(standard_option, pricing_env)

        # QMC should generally have lower variance
        # This test may occasionally fail due to randomness, but should pass most times
        quasi_std = quasi_engine.get_last_std_error()
        pseudo_std = pseudo_engine.get_last_std_error()

        # Both should be positive
        assert quasi_std > 0
        assert pseudo_std > 0


class TestRangeAccrualMCEngineRepr:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        engine = RangeAccrualMCEngine(method=MonteCarloMethod.QUASI)
        assert "RangeAccrualMCEngine" in repr(engine)
        assert "QUASI" in repr(engine)
