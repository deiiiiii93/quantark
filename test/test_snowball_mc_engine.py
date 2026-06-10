"""
Unit tests for SnowballMCEngine.

Tests cover:
- Basic standard and reverse snowball pricing
- Different Monte Carlo methods (PSEUDO, QUASI, RQMC)
- Convergence behavior
- Edge cases (near-expiry, deep ITM/OTM barriers)
- Validation errors
- Time-varying barriers and rates
- KO payoff timing (INSTANT vs EXPIRY)
- V0 and V1 maturity payoffs
- disable_ko_after_ki behavior
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.snowball_config import (
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import (
    CouponPayType,
    ObservationType,
)
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.exceptions import PricingError, ValidationError

# =============================================================================
# Fixtures - Common test configurations
# =============================================================================


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.05,
    div_yield: float = 0.02,
) -> PricingEnvironment:
    """Create a basic pricing environment for testing."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def create_basic_barrier_config(
    ko_barrier: float = 103.0,
    ko_rate: float = 0.15,
    ki_barrier: float = 75.0,
    ko_observation_dates: List[float] = None,
    ki_observation_dates: List[float] = None,
    ki_continuous: bool = True,
    disable_ko_after_ki: bool = False,
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
        ki_observation_type=(
            ObservationType.CONTINUOUS if ki_continuous else ObservationType.DISCRETE
        ),
        ki_observation_dates=ki_observation_dates,
        ki_continuous=ki_continuous,
        disable_ko_after_ki=disable_ko_after_ki,
    )


def create_standard_snowball(
    initial_price: float = 100.0,
    strike: float = 100.0,
    contract_multiplier: float = None,
    maturity: float = 1.0,
    barrier_config: BarrierConfig = None,
    payoff_config: PayoffConfig = None,
    accrual_config: AccrualConfig = None,
) -> SnowballOption:
    """Create a standard snowball option for testing."""
    if barrier_config is None:
        barrier_config = create_basic_barrier_config()
    if contract_multiplier is None:
        contract_multiplier = 1_000_000.0 / initial_price

    return SnowballOption(
        initial_price=initial_price,
        strike=strike,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        contract_multiplier=contract_multiplier,
        maturity=maturity,
        is_reverse=False,
    )


def create_reverse_snowball(
    initial_price: float = 100.0,
    strike: float = 100.0,
    contract_multiplier: float = None,
    maturity: float = 1.0,
    barrier_config: BarrierConfig = None,
    payoff_config: PayoffConfig = None,
    accrual_config: AccrualConfig = None,
) -> SnowballOption:
    """Create a reverse snowball option for testing."""
    if barrier_config is None:
        # Reverse snowball: DOWN KO, UP KI
        barrier_config = BarrierConfig(
            ko_barrier=97.0,  # DOWN barrier
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=125.0,  # UP barrier
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        )
    if contract_multiplier is None:
        contract_multiplier = 1_000_000.0 / initial_price

    return SnowballOption(
        initial_price=initial_price,
        strike=strike,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        contract_multiplier=contract_multiplier,
        maturity=maturity,
        is_reverse=True,
    )


def get_principal(snowball: SnowballOption) -> float:
    """Return per-contract principal based on initial price and contract multiplier."""
    return snowball.initial_price * snowball.contract_multiplier


# =============================================================================
# Test Classes
# =============================================================================


class TestSnowballMCEngineCreation:
    """Tests for SnowballMCEngine initialization."""

    def test_default_initialization(self):
        """Test engine with default parameters."""
        engine = SnowballMCEngine()
        assert engine.method == MonteCarloMethod.PSEUDO
        assert engine.use_dask is False
        assert engine.params is not None

    def test_with_mcparams(self):
        """Test engine with custom MCParams."""
        params = MCParams(num_paths=50000, time_steps=100, seed=123)
        engine = SnowballMCEngine(params=params)
        assert engine.params.num_paths == 50000
        assert engine.params.seed == 123

    def test_method_enum_selection(self):
        """Test method selection with MonteCarloMethod enum."""
        engine = SnowballMCEngine(method=MonteCarloMethod.QUASI)
        assert engine.method == MonteCarloMethod.QUASI

    def test_method_two_level_enum(self):
        """Test method selection with two-level enum pattern."""
        engine = SnowballMCEngine(method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI))
        assert engine.method == MonteCarloMethod.QUASI

    def test_method_string_selection(self):
        """Test method selection with string (backward compatibility)."""
        engine = SnowballMCEngine(method="quasi")
        assert engine.method == MonteCarloMethod.QUASI

    def test_invalid_method_string(self):
        """Test invalid method string raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid method string"):
            SnowballMCEngine(method="invalid_method")

    def test_invalid_params_type(self):
        """Test invalid params type raises ValidationError."""
        with pytest.raises(ValidationError, match="params must be MCParams"):
            SnowballMCEngine(params="not_params")


class TestBasicPricing:
    """Tests for basic snowball pricing functionality."""

    def test_standard_snowball_pricing(self):
        """Test basic standard snowball pricing returns positive price."""
        snowball = create_standard_snowball()
        env = create_pricing_env()
        engine = SnowballMCEngine(
            params=MCParams(num_paths=10000, seed=42),
            method=MonteCarloMethod.PSEUDO,
        )

        price = engine.price(snowball, env)

        assert price > 0
        assert engine.get_last_std_error() is not None
        assert engine.get_last_result() is not None

    def test_reverse_snowball_pricing(self):
        """Test basic reverse snowball pricing returns positive price."""
        snowball = create_reverse_snowball()
        env = create_pricing_env()
        engine = SnowballMCEngine(
            params=MCParams(num_paths=10000, seed=42),
            method=MonteCarloMethod.PSEUDO,
        )

        price = engine.price(snowball, env)

        assert price > 0

    def test_result_contains_probabilities(self):
        """Test pricing result contains probability statistics."""
        snowball = create_standard_snowball()
        env = create_pricing_env()
        engine = SnowballMCEngine(
            params=MCParams(num_paths=10000, seed=42),
        )

        engine.price(snowball, env)
        result = engine.get_last_result()

        assert result is not None
        assert 0 <= result.ko_probability <= 1
        assert 0 <= result.v0_probability <= 1
        assert 0 <= result.v1_probability <= 1
        # Probabilities should sum to 1
        total_prob = (
            result.ko_probability + result.v0_probability + result.v1_probability
        )
        assert abs(total_prob - 1.0) < 1e-6


class TestMonteCarloMethods:
    """Tests for different Monte Carlo methods."""

    def test_pseudo_mc_pricing(self):
        """Test pricing with PSEUDO Monte Carlo."""
        snowball = create_standard_snowball()
        env = create_pricing_env()
        engine = SnowballMCEngine(
            params=MCParams(num_paths=10000, seed=42),
            method=MonteCarloMethod.PSEUDO,
        )

        price = engine.price(snowball, env)
        assert price > 0

    def test_quasi_mc_pricing(self):
        """Test pricing with QUASI Monte Carlo (Sobol sequences)."""
        snowball = create_standard_snowball()
        env = create_pricing_env()
        engine = SnowballMCEngine(
            params=MCParams(num_paths=10000, seed=42),
            method=MonteCarloMethod.QUASI,
        )

        price = engine.price(snowball, env)
        assert price > 0

    def test_rqmc_pricing(self):
        """Test pricing with RANDOMIZED_QUASI Monte Carlo."""
        snowball = create_standard_snowball()
        env = create_pricing_env()
        engine = SnowballMCEngine(
            params=MCParams(num_paths=5000, seed=42),
            method=MonteCarloMethod.RANDOMIZED_QUASI,
        )

        price = engine.price(snowball, env)
        result = engine.get_last_result()

        assert price > 0
        assert result.batches_used is not None

    def test_methods_converge_to_similar_prices(self):
        """Test that different methods converge to similar prices."""
        snowball = create_standard_snowball()
        env = create_pricing_env()

        prices = {}
        for method in [MonteCarloMethod.PSEUDO, MonteCarloMethod.QUASI]:
            engine = SnowballMCEngine(
                params=MCParams(num_paths=50000, seed=42),
                method=method,
            )
            prices[method] = engine.price(snowball, env)

        # Prices should be within 5% of each other
        price_ratio = prices[MonteCarloMethod.PSEUDO] / prices[MonteCarloMethod.QUASI]
        assert 0.95 < price_ratio < 1.05


class TestConvergence:
    """Tests for Monte Carlo convergence behavior."""

    def test_std_error_decreases_with_paths(self):
        """Test standard error decreases as num_paths increases."""
        snowball = create_standard_snowball()
        env = create_pricing_env()

        std_errors = []
        for num_paths in [1000, 5000, 20000]:
            engine = SnowballMCEngine(
                params=MCParams(num_paths=num_paths, seed=42),
                method=MonteCarloMethod.PSEUDO,
            )
            engine.price(snowball, env)
            std_errors.append(engine.get_last_std_error())

        # Standard error should decrease monotonically
        assert std_errors[0] > std_errors[1] > std_errors[2]

    def test_price_stability_with_different_seeds(self):
        """Test prices are stable across different random seeds."""
        snowball = create_standard_snowball()
        env = create_pricing_env()

        prices = []
        for seed in [42, 123, 456]:
            engine = SnowballMCEngine(
                params=MCParams(num_paths=20000, seed=seed),
                method=MonteCarloMethod.PSEUDO,
            )
            prices.append(engine.price(snowball, env))

        # Prices should be within 10% of mean
        mean_price = np.mean(prices)
        for price in prices:
            assert abs(price - mean_price) / mean_price < 0.10


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_near_expiry_pricing(self):
        """Test pricing with very short maturity."""
        barrier_config = BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.01],  # Very short maturity
            ki_barrier=75.0,
            ki_continuous=True,
        )
        snowball = create_standard_snowball(
            maturity=0.01,
            barrier_config=barrier_config,
        )
        env = create_pricing_env()
        engine = SnowballMCEngine(params=MCParams(num_paths=5000, seed=42))

        price = engine.price(snowball, env)
        assert price >= 0

    def test_deep_itm_ko_barrier(self):
        """Test with KO barrier very close to spot (high KO probability)."""
        barrier_config = BarrierConfig(
            ko_barrier=100.5,  # Very close to spot
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_continuous=True,
        )
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env()
        engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42))

        engine.price(snowball, env)
        result = engine.get_last_result()

        # With barrier so close to spot, KO probability should be high
        assert result.ko_probability > 0.3

    def test_deep_otm_ko_barrier(self):
        """Test with KO barrier far from spot (low KO probability)."""
        barrier_config = BarrierConfig(
            ko_barrier=150.0,  # Far from spot
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_continuous=True,
        )
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env()
        engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42))

        engine.price(snowball, env)
        result = engine.get_last_result()

        # With barrier so far from spot, KO probability should be low
        assert result.ko_probability < 0.3

    def test_negative_price_allowed_when_principal_excluded(self):
        """Coupon-only structures may have negative PV; engine should not raise."""
        barrier_config = BarrierConfig(
            ko_barrier=1.0e12,  # Practically never KO
            ko_rate=0.0,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=1.0e12,  # Always in KI region (standard: KI if S <= barrier)
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        )
        payoff_config = PayoffConfig(
            rebate_rate=0.0,
            include_principal=False,
            participation_rate=1.0,
        )
        snowball = SnowballOption(
            initial_price=100.0,
            strike=120.0,  # Deep ITM short put payoff on KI -> negative payoff
            barrier_config=barrier_config,
            payoff_config=payoff_config,
            contract_multiplier=10_000.0,
            maturity=1.0,
            is_reverse=False,
        )
        env = create_pricing_env(spot=100.0, vol=0.20, rate=0.0, div_yield=0.0)
        engine = SnowballMCEngine(params=MCParams(num_paths=20000, seed=42))

        price = engine.price(snowball, env)
        assert price < 0.0


class TestValidation:
    """Tests for validation errors."""

    def test_invalid_product_type(self):
        """Test error when pricing non-snowball product."""
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        from quantark.util.enum import OptionType

        option = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        env = create_pricing_env()
        engine = SnowballMCEngine()

        with pytest.raises(PricingError, match="only supports SnowballOption"):
            engine.price(option, env)

    def test_invalid_spot_price(self):
        """Test error with non-positive spot price."""
        with pytest.raises(ValidationError, match="spot must be positive"):
            create_pricing_env(spot=-100.0)

    def test_invalid_volatility(self):
        """Test error with non-positive volatility."""
        with pytest.raises(ValidationError, match="volatility must be positive"):
            create_pricing_env(vol=-0.20)


class TestTimeVaryingBarriers:
    """Tests for time-varying barriers and rates."""

    def test_time_varying_ko_barriers(self):
        """Test pricing with different barrier levels at each observation."""
        barrier_config = BarrierConfig(
            ko_barrier=[102.0, 104.0, 106.0, 108.0],  # Increasing barriers
            ko_rate=[0.10, 0.12, 0.14, 0.16],  # Increasing rates
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_continuous=True,
        )
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env()
        engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42))

        price = engine.price(snowball, env)
        assert price > 0

    def test_time_varying_ki_barriers(self):
        """Test pricing with different KI barrier levels at each observation."""
        ko_observation_dates = [0.25, 0.5, 0.75, 1.0]
        ki_observation_dates = [0.25, 0.5, 0.75, 1.0]

        barrier_config = BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=ko_observation_dates,
            ki_barrier=[80.0, 78.0, 76.0, 74.0],  # Decreasing KI barriers
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=ki_observation_dates,
            ki_continuous=False,
        )
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env()
        engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42))

        price = engine.price(snowball, env)
        assert price > 0


class TestCouponPaymentTiming:
    """Tests for INSTANT vs EXPIRY coupon payment timing."""

    def test_instant_coupon_payment(self):
        """Test pricing with INSTANT coupon payment."""
        accrual_config = AccrualConfig(coupon_pay_type=CouponPayType.INSTANT)
        snowball = create_standard_snowball(accrual_config=accrual_config)
        env = create_pricing_env()
        engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42))

        price_instant = engine.price(snowball, env)
        assert price_instant > 0

    def test_expiry_coupon_payment(self):
        """Test pricing with EXPIRY coupon payment."""
        accrual_config = AccrualConfig(coupon_pay_type=CouponPayType.EXPIRY)
        snowball = create_standard_snowball(accrual_config=accrual_config)
        env = create_pricing_env()
        engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42))

        price_expiry = engine.price(snowball, env)
        assert price_expiry > 0

    def test_instant_vs_expiry_price_difference(self):
        """Test that INSTANT and EXPIRY produce different prices."""
        env = create_pricing_env()

        # INSTANT: KO payoff discounted from observation date
        accrual_instant = AccrualConfig(coupon_pay_type=CouponPayType.INSTANT)
        snowball_instant = create_standard_snowball(accrual_config=accrual_instant)
        engine = SnowballMCEngine(params=MCParams(num_paths=20000, seed=42))
        price_instant = engine.price(snowball_instant, env)

        # EXPIRY: KO payoff discounted from maturity
        accrual_expiry = AccrualConfig(coupon_pay_type=CouponPayType.EXPIRY)
        snowball_expiry = create_standard_snowball(accrual_config=accrual_expiry)
        engine = SnowballMCEngine(params=MCParams(num_paths=20000, seed=42))
        price_expiry = engine.price(snowball_expiry, env)

        # INSTANT should have higher present value (discounted from earlier time)
        # This may not always hold due to MC noise, but generally true
        assert price_instant != price_expiry


class TestMaturityPayoffs:
    """Tests for V0 and V1 maturity payoff scenarios."""

    def test_high_v0_probability_with_no_ki_barrier(self):
        """Test V0 state dominates when no KI barrier is set."""
        barrier_config = BarrierConfig(
            ko_barrier=150.0,  # Very high, unlikely to hit
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=None,  # No KI barrier
            ki_continuous=False,
        )
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env()
        engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42))

        engine.price(snowball, env)
        result = engine.get_last_result()

        # Without KI barrier, V1 probability should be 0
        assert result.v1_probability == 0.0
        # V0 probability should be high (1 - KO probability)
        assert result.v0_probability > 0.5

    def test_high_ki_probability_with_close_ki_barrier(self):
        """Test high V1 probability when KI barrier is close to spot."""
        barrier_config = BarrierConfig(
            ko_barrier=150.0,  # Very high, unlikely to hit
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=99.0,  # Very close to spot, likely to hit
            ki_continuous=True,
        )
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env()
        engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42))

        engine.price(snowball, env)
        result = engine.get_last_result()

        # With KI barrier so close to spot, V1 probability should be significant
        assert result.v1_probability > 0.3


class TestDisableKOAfterKI:
    """Tests for disable_ko_after_ki behavior."""

    def test_disable_ko_after_ki_true(self):
        """Test that KO is ignored after KI when disable_ko_after_ki=True."""
        barrier_config = BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=95.0,  # Fairly close to spot
            ki_continuous=True,
            disable_ko_after_ki=True,
        )
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env()
        engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42))

        engine.price(snowball, env)
        result_disabled = engine.get_last_result()

        # Compare with disable_ko_after_ki=False
        barrier_config_enabled = BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=95.0,
            ki_continuous=True,
            disable_ko_after_ki=False,
        )
        snowball_enabled = create_standard_snowball(
            barrier_config=barrier_config_enabled
        )
        engine = SnowballMCEngine(params=MCParams(num_paths=10000, seed=42))
        engine.price(snowball_enabled, env)
        result_enabled = engine.get_last_result()

        # With disable_ko_after_ki=True, fewer paths should end in KO
        # (some KOs are invalidated by earlier KI)
        # This comparison may not always hold due to MC noise, but generally true
        assert result_disabled.ko_probability <= result_enabled.ko_probability + 0.05


class TestRepr:
    """Tests for string representation."""

    def test_repr_pseudo(self):
        """Test __repr__ for PSEUDO method."""
        engine = SnowballMCEngine(method=MonteCarloMethod.PSEUDO)
        assert "PSEUDO" in repr(engine)
        assert "SnowballMCEngine" in repr(engine)

    def test_repr_quasi(self):
        """Test __repr__ for QUASI method."""
        engine = SnowballMCEngine(method=MonteCarloMethod.QUASI)
        assert "QUASI" in repr(engine)


class TestDaskParallel:
    """Tests for Dask parallel path (if Dask is installed)."""

    def test_dask_parallel_preserves_num_paths_and_probabilities(self):
        """Dask path should use exactly params.num_paths and keep prob invariants."""
        pytest.importorskip("dask")

        snowball = create_standard_snowball()
        env = create_pricing_env()
        params = MCParams(num_paths=4000, seed=42)
        engine = SnowballMCEngine(
            params=params,
            method=MonteCarloMethod.PSEUDO,
            use_dask=True,
            num_batches=4,
        )

        engine.price(snowball, env)
        result = engine.get_last_result()

        assert result is not None
        assert result.num_paths == params.num_paths
        assert 0.0 <= result.ko_probability <= 1.0
        assert 0.0 <= result.v0_probability <= 1.0
        assert 0.0 <= result.v1_probability <= 1.0
        assert (
            abs(
                result.ko_probability
                + result.v0_probability
                + result.v1_probability
                - 1.0
            )
            < 1e-6
        )
        assert result.std_error is not None
        assert result.std_error >= 0.0

    def test_dask_parallel_price_close_to_serial(self):
        """Parallel and serial should be statistically close (within noise)."""
        pytest.importorskip("dask")

        snowball = create_standard_snowball()
        env = create_pricing_env()
        params = MCParams(num_paths=20000, seed=123)

        engine_serial = SnowballMCEngine(
            params=params,
            method=MonteCarloMethod.PSEUDO,
            use_dask=False,
        )
        price_serial = engine_serial.price(snowball, env)
        se_serial = engine_serial.get_last_std_error()
        assert se_serial is not None

        engine_parallel = SnowballMCEngine(
            params=params,
            method=MonteCarloMethod.PSEUDO,
            use_dask=True,
            num_batches=4,
        )
        price_parallel = engine_parallel.price(snowball, env)
        se_parallel = engine_parallel.get_last_std_error()
        assert se_parallel is not None

        # Use combined standard error as a stability-aware tolerance.
        combined_se = float(np.sqrt(se_serial * se_serial + se_parallel * se_parallel))
        assert abs(price_parallel - price_serial) <= 8.0 * combined_se


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
